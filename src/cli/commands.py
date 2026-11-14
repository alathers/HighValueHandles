"""CLI commands for bourbon image processor.

Provides command-line interface for processing bourbon bottle images
using argparse for argument parsing and structured command execution.
"""

import argparse
import logging
import sys
from pathlib import Path

from image_processor.batch_processor import BatchProcessor
from image_processor.bottle_analyzer import BottleAnalyzer
from image_processor.image_organizer import ImageOrganizer
from image_processor.preview_generator import PreviewGenerator
from lib.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of arguments (default: sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Process bourbon bottle images and extract metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and verbose output",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process command (default behavior)
    process_parser = subparsers.add_parser(
        "process",
        help="Process bourbon bottle images and extract metadata",
    )

    # Input/output options for process command
    process_parser.add_argument(
        "-f",
        "--image-folder",
        type=Path,
        default=Path(),
        help="Folder containing images to process (default: current directory)",
    )
    process_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for processed bottles (default: processed/)",
    )

    # Processing options
    process_parser.add_argument(
        "-b",
        "--batch-size",
        type=str,
        default="5",
        help="Number of bottles to process per batch, or 'all' to process all (default: 5)",
    )
    process_parser.add_argument(
        "--lookahead",
        type=int,
        default=3,
        help="Number of images to peek ahead for multi-angle photos (default: 3)",
    )
    process_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent operations (default: 3)",
    )

    # Model options
    process_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen2-vl:7b",
        help="Ollama vision model to use for analysis (default: qwen2-vl:7b)",
    )
    process_parser.add_argument(
        "-u",
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama service URL (default: http://localhost:11434)",
    )

    # Behavior options
    process_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enable interactive preview and review mode",
    )
    process_parser.add_argument(
        "-v",
        "--include-vendor-images",
        action="store_true",
        help="Include vendor/supply chain images in processing",
    )
    process_parser.add_argument(
        "-r",
        "--allow-retry",
        action="store_true",
        default=True,
        help="Allow automatic retry on processing failures (default: True)",
    )
    process_parser.add_argument(
        "--no-retry",
        action="store_false",
        dest="allow_retry",
        help="Disable automatic retry on processing failures",
    )

    # Debug options for process command
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actually processing images (for testing)",
    )

    # Feedback command
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Apply feedback and corrections to processed batch"
    )

    feedback_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory containing processed batches (default: processed/)",
    )
    feedback_parser.add_argument(
        "--batch",
        type=str,
        help="Batch ID to apply feedback to (e.g., '1' for batch_1_preview.md)",
    )
    feedback_parser.add_argument(
        "--corrections",
        type=Path,
        help="Path to JSON file containing corrections",
    )
    feedback_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enable interactive feedback collection mode",
    )

    # Retry command
    retry_parser = subparsers.add_parser(
        "retry",
        help="Retry processing failed bottles from error reports"
    )

    retry_parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to error_report.json file",
    )
    retry_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for processed bottles (default: processed/)",
    )

    return parser.parse_args(argv)


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag.

    Args:
        debug: Whether to enable DEBUG-level logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        stream=sys.stdout,
    )


def pre_flight_checks(args: argparse.Namespace) -> int:
    """Verify prerequisites before processing.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 if checks pass, non-zero if checks fail.
    """
    # Validate argument values first
    if args.command == "process":
        # Validate batch_size: integer >= 1 OR literal 'all'
        batch_size = args.batch_size
        if batch_size.lower() != "all":
            try:
                batch_int = int(batch_size)
                if batch_int < 1:
                    logger.error("Invalid --batch-size: must be >= 1 or 'all', got '%s'", batch_size)
                    return 1
            except ValueError:
                logger.error("Invalid --batch-size: must be integer or 'all', got '%s'", batch_size)
                return 1

        # Validate lookahead: integer 0-10
        if args.lookahead < 0 or args.lookahead > 10:
            logger.error("Invalid --lookahead: must be between 0 and 10, got '%s'", args.lookahead)
            return 1

        # Validate concurrency: integer 1-10
        if args.concurrency < 1 or args.concurrency > 10:
            logger.error("Invalid --concurrency: must be between 1 and 10, got '%s'", args.concurrency)
            return 1

        # Validate ollama_url format
        from urllib.parse import urlparse

        try:
            parsed = urlparse(args.ollama_url)
            if not parsed.scheme or not parsed.netloc:
                logger.error("Invalid --ollama-url: must be valid URL, got '%s'", args.ollama_url)
                return 1
        except Exception as e:
            logger.error("Invalid --ollama-url: %s", e)
            return 1

    # Check input folder exists
    if not args.image_folder.exists():
        logger.error("Input folder does not exist: %s", args.image_folder)
        return 3

    if not args.image_folder.is_dir():
        logger.error("Input path is not a directory: %s", args.image_folder)
        return 3

    # Check folder contains supported images
    image_extensions = {".jpeg", ".jpg", ".png", ".tiff", ".heic"}
    images = list(args.image_folder.rglob("*"))
    image_files = [p for p in images if p.suffix.lower() in image_extensions]

    if not image_files:
        logger.error("No supported images found in folder: %s", args.image_folder)
        logger.info("Supported formats: .jpeg, .jpg, .png, .tiff, .heic")
        return 3

    logger.info("Found %s images in %s", len(image_files), args.image_folder)

    # Check Ollama service reachable
    try:
        ollama_client = OllamaClient(base_url=args.ollama_url)
        if not ollama_client.health_check():
            logger.error("Ollama service not reachable at %s", args.ollama_url)
            return 2
        logger.info("Ollama service is reachable")
    except Exception:
        logger.exception("Failed to connect to Ollama")
        return 2

    # Check model available
    try:
        ollama_client = OllamaClient(base_url=args.ollama_url)
        if not ollama_client.is_model_available(args.model):
            logger.error("Model '%s' is not available in Ollama", args.model)
            logger.info("Available models:")
            models = ollama_client.list_models()
            if models:
                for model in models:
                    logger.info("  - %s", model)
            return 2
        logger.info("Model '%s' is available", args.model)
    except Exception:
        logger.exception("Failed to check model availability")
        return 2

    # Check Tesseract OCR installed
    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR version: %s", version)
    except Exception:
        logger.exception("Tesseract OCR is not installed or not found in PATH")
        logger.info("Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        return 1

    # Check output directory writable
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        test_file = args.output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        logger.info("Output directory is writable: %s", args.output_dir)
    except Exception:
        logger.exception("Output directory is not writable: %s", args.output_dir)
        return 5

    return 0


def process_command(args: argparse.Namespace) -> int:
    """Process bourbon bottle images from input folder.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 (success), 1 (general error), 2 (Ollama connection failed),
                   3 (no valid images found), 4 (all bottles failed), 5 (output directory creation failed).
    """
    # Run pre-flight checks
    exit_code = pre_flight_checks(args)
    if exit_code != 0:
        return exit_code

    # Initialize components
    ollama_client = OllamaClient(base_url=args.ollama_url)
    image_organizer = ImageOrganizer()
    bottle_analyzer = BottleAnalyzer(ollama_client=ollama_client, image_organizer=image_organizer)
    batch_processor = BatchProcessor(
        batch_size=args.batch_size,
        lookahead=args.lookahead,
        bottle_analyzer=bottle_analyzer,
        similarity_threshold=0.7,
    )
    preview_generator = PreviewGenerator(output_dir=args.output_dir)

    # Scan images using ImageOrganizer
    images = image_organizer.scan_images(args.image_folder, recursive=True)

    logger.info("Processing %s images...", len(images))

    if args.dry_run:
        logger.info("Dry-run mode: would process images but not execute")
        logger.info("Images: %s", [img.path for img in images])
        return 0

    # Group images by bottle using ImageOrganizer
    image_groups = image_organizer.group_by_bottle(images)
    logger.info("Found %s bottle groups", len(image_groups))

    # Process images using BatchProcessor
    processed_count = 0
    failed_count = 0
    error_count = 0

    # Slice image groups into batches
    batches = batch_processor.slice_into_batches(image_groups)

    # Process each batch
    for batch_idx, batch in enumerate(batches, 1):
        logger.info("Processing batch %s/%s (%s image groups)", batch_idx, len(batches), len(batch))

        # Apply lookahead to include multi-angle photos
        if batch_idx < len(batches):
            remaining_batches = batches[batch_idx:]
            remaining_groups = [group for batch in remaining_batches for group in batch]
            augmented_batch, updated_remaining = batch_processor.apply_lookahead(batch, remaining_groups)
            logger.info("Lookahead: batch expanded from %s to %s groups", len(batch), len(augmented_batch))
            batch = augmented_batch

        # Process the batch
        batch_result = batch_processor.process_batch(
            batch,
            vision_model=args.model,
        )

        # Generate batch preview report (skip if processing all in single batch)
        if str(args.batch_size).lower() != "all":
            try:
                batch_id = f"{batch_idx}"
                report_markdown = preview_generator.generate_batch_preview_report(
                    batch_result.bottle_dicts,
                    batch_result.metadata_list,
                    batch_result.failed_bottles,
                    batch_id,
                )
                preview_path = preview_generator.save_batch_preview(
                    batch_result.bottle_dicts,
                    batch_result.metadata_list,
                    batch_id,
                    args.output_dir,
                )
                logger.info("Saved batch preview report to %s", preview_path)
            except Exception:
                logger.exception("Failed to generate batch preview report")

        # Interactive mode: prompt user for corrections
        if args.interactive:
            try:
                from models.bottle import Correction

                correction_dicts, rejected_bottles_from_prompt = preview_generator.interactive_prompt(
                    batch_result.bottle_dicts,
                    batch_result.metadata_list,
                )

                if correction_dicts:
                    # Convert dict corrections to Correction objects
                    corrections = [
                        Correction(
                            field_name=cd["field_name"],
                            original_value=cd["original_value"],
                            corrected_value=cd["corrected_value"],
                        )
                        for cd in correction_dicts
                    ]

                    # Apply corrections to bottle_dicts
                    batch_result.bottle_dicts = batch_processor.apply_corrections(
                        batch_result.bottle_dicts,
                        corrections,
                    )

                    # Learn from corrections for future batches
                    batch_processor.learn_from_feedback(corrections)

                    logger.info("Applied %d corrections from interactive feedback", len(corrections))

                # Add rejected bottles to failed list
                batch_result.failed_bottles.extend(rejected_bottles_from_prompt)
                if rejected_bottles_from_prompt:
                    logger.info("Rejected %d bottles from interactive feedback", len(rejected_bottles_from_prompt))

            except Exception:
                logger.exception("Interactive correction failed")

        # Save each successfully processed bottle
        for bottle_dict, metadata in zip(batch_result.bottle_dicts, batch_result.metadata_list):
            try:
                # Detect quantity from metadata's source images
                image_list = [
                    next((img for img in batch[0] if img.path in metadata.source_images), None)
                    for metadata in [metadata]
                ]

                if image_list and image_list[0]:
                    quantity = bottle_analyzer.detect_quantity([image_list[0]])
                else:
                    quantity = 1

                # Create Bottle object from dict
                from models.bottle import Bottle

                bottle = Bottle(
                    name=bottle_dict.get("name", "Unknown"),
                    distillery=bottle_dict.get("distillery", "Unknown"),
                    msrp=float(bottle_dict.get("msrp", "0.00")),
                    specific_release=bottle_dict.get("specific_release", ""),
                    proof=bottle_dict.get("proof"),
                    age=bottle_dict.get("age"),
                    tasting_notes=bottle_dict.get("tasting_notes"),
                    quantity=quantity,
                )

                # Save output files
                output_folder = preview_generator.save_bottle_files(bottle, metadata)
                logger.info("Saved bottle to %s", output_folder)

                processed_count += 1

            except Exception:
                logger.exception("Failed to save bottle output")
                failed_count += 1

        # Handle failed bottles in batch
        for failed_bottle_id in batch_result.failed_bottles:
            logger.warning("Bottle failed to process: %s", failed_bottle_id)
            failed_count += 1

    # Completion reporting
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info("Processed successfully: %s", processed_count)
    logger.info("Failed: %s", failed_count)
    logger.info("Error reports generated: %s", error_count)
    logger.info("Output directory: %s", args.output_dir)

    if failed_count == 0:
        return 0
    if processed_count == 0:
        logger.error("All bottles failed to process")
        return 4
    logger.warning("Some bottles failed (%s/%s)", failed_count, len(image_groups))
    return 1


def feedback_command(args: argparse.Namespace) -> int:
    """Apply feedback and corrections to processed batch.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 (success), 1 (general error), 2 (batch not found).
    """
    from models.bottle import Bottle, Correction

    # Determine batch to load
    if args.batch:
        batch_preview_file = args.output_dir / f"batch_{args.batch}_preview.md"
        batch_json_file = args.output_dir / f"batch_{args.batch}.json"
    else:
        # Find most recent batch preview
        batch_preview_files = sorted(args.output_dir.glob("batch_*_preview.md"))
        if not batch_preview_files:
            logger.error("No batch preview files found in %s", args.output_dir)
            return 2
        batch_preview_files.sort(reverse=True)
        batch_preview_file = batch_preview_files[0]
        batch_json_file = batch_preview_file.with_suffix(".json")

    logger.info("Using batch file: %s", batch_preview_file)

    # Load batch data from JSON if available
    bottle_dicts = []
    metadata_list = []

    if batch_json_file.exists():
        try:
            import json

            with batch_json_file.open("r", encoding="utf-8") as f:
                batch_data = json.load(f)
                bottle_dicts = batch_data.get("bottles", [])
                # Reconstruct metadata from loaded data
                from models.bottle import ExtractedMetadata

                metadata_list = [
                    ExtractedMetadata.from_dict(m) for m in batch_data.get("metadata", [])
                ]
            logger.info("Loaded %d bottles from batch JSON", len(bottle_dicts))
        except Exception:
            logger.exception("Failed to load batch JSON file")

    # Load corrections from JSON file if provided
    corrections: list[Correction] = []

    if args.corrections and args.corrections.exists():
        try:
            import json

            with args.corrections.open("r", encoding="utf-8") as f:
                corrections_data = json.load(f)
                corrections = [Correction.from_dict(c) for c in corrections_data]
            logger.info("Loaded %d corrections from %s", len(corrections), args.corrections)
        except Exception:
            logger.exception("Failed to load corrections file")
            return 1

    # Apply corrections
    if corrections and bottle_dicts:
        batch_processor = BatchProcessor()
        batch_processor.bottle_analyzer = BottleAnalyzer()

        modified_bottles = batch_processor.apply_corrections(bottle_dicts, corrections)
        logger.info("Applied %d corrections to %d bottles", len(corrections), len(modified_bottles))

        # Save corrected bottles
        for bottle_dict, metadata in zip(modified_bottles, metadata_list):
            try:
                bottle = Bottle(
                    name=bottle_dict.get("name", "Unknown"),
                    distillery=bottle_dict.get("distillery", "Unknown"),
                    msrp=float(bottle_dict.get("msrp", "0.00")),
                    specific_release=bottle_dict.get("specific_release", ""),
                    proof=bottle_dict.get("proof"),
                    age=bottle_dict.get("age"),
                    tasting_notes=bottle_dict.get("tasting_notes"),
                    metadata=metadata,
                    corrections=[Correction.from_dict(c) for c in bottle_dict.get("corrections", [])],
                )

                # Resave bottle files with corrections
                from image_processor.preview_generator import PreviewGenerator

                preview_generator = PreviewGenerator(output_dir=args.output_dir)
                output_folder = preview_generator.save_bottle_files(bottle, metadata)
                logger.info("Resaved bottle to %s", output_folder)

            except Exception:
                logger.exception("Failed to resave bottle")

        logger.info("Feedback processing complete")
        return 0

    # Interactive mode
    if args.interactive:
        try:
            from image_processor.preview_generator import PreviewGenerator

            preview_generator = PreviewGenerator(output_dir=args.output_dir)
            batch_processor = BatchProcessor()

            correction_dicts, rejected_bottles = preview_generator.interactive_prompt(
                bottle_dicts,
                metadata_list,
            )

            if correction_dicts:
                # Convert dict corrections to Correction objects
                corrections = [
                    Correction(
                        field_name=cd["field_name"],
                        original_value=cd["original_value"],
                        corrected_value=cd["corrected_value"],
                    )
                    for cd in correction_dicts
                ]

                # Apply corrections to bottle_dicts
                modified_bottles = batch_processor.apply_corrections(bottle_dicts, corrections)

                # Learn from corrections
                batch_processor.learn_from_feedback(corrections)

                logger.info("Applied %d corrections from interactive feedback", len(corrections))

                # Save corrected bottles
                for bottle_dict, metadata in zip(modified_bottles, metadata_list):
                    try:
                        bottle = Bottle(
                            name=bottle_dict.get("name", "Unknown"),
                            distillery=bottle_dict.get("distillery", "Unknown"),
                            msrp=float(bottle_dict.get("msrp", "0.00")),
                            specific_release=bottle_dict.get("specific_release", ""),
                            proof=bottle_dict.get("proof"),
                            age=bottle_dict.get("age"),
                            tasting_notes=bottle_dict.get("tasting_notes"),
                            metadata=metadata,
                            corrections=corrections,
                        )

                        output_folder = preview_generator.save_bottle_files(bottle, metadata)
                        logger.info("Resaved bottle to %s", output_folder)

                    except Exception:
                        logger.exception("Failed to resave bottle")

            logger.info("Interactive feedback complete")
            return 0

        except Exception:
            logger.exception("Interactive feedback failed")
            return 1

    logger.info("No corrections applied")
    return 0


def retry_command(args: argparse.Namespace) -> int:
    """Retry processing failed bottles from error report.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 (success), 1 (general error), 2 (report not found).
    """
    from models.bottle import Bottle
    from models.image import Image

    # Check error report exists
    if not args.report.exists():
        logger.error("Error report not found: %s", args.report)
        return 2

    # Load error report
    import json

    try:
        with args.report.open("r", encoding="utf-8") as f:
            error_report = json.load(f)
    except Exception:
        logger.exception("Failed to read error report")
        return 1

    logger.info("Loading error report: %s", args.report)

    # Validate error report structure
    required_fields = ["bottle_id", "images", "error_message"]
    for field in required_fields:
        if field not in error_report:
            logger.error("Missing required field in error report: %s", field)
            return 1

    bottle_id = error_report["bottle_id"]
    image_paths = error_report["images"]
    error_message = error_report["error_message"]
    suggested_fix = error_report.get("suggested_fix", {})

    logger.info("Retrying bottle: %s", bottle_id)
    logger.info("Error was: %s", error_message)
    if suggested_fix:
        logger.info("Suggested fix: %s", suggested_fix)

    # Convert image paths to Image objects
    images = []
    for img_path in image_paths:
        path = Path(img_path)
        if path.exists():
            images.append(Image(path=str(path)))
        else:
            logger.warning("Image not found: %s", img_path)

    if not images:
        logger.error("No valid images found in error report")
        return 1

    # Initialize components for re-processing
    ollama_client = OllamaClient(base_url="http://localhost:11434")
    image_organizer = ImageOrganizer()
    bottle_analyzer = BottleAnalyzer(ollama_client=ollama_client, image_organizer=image_organizer)

    # Apply suggested corrections if provided
    corrections: list = []
    if suggested_fix:
        from models.bottle import Correction

        for field_name, correction_value in suggested_fix.items():
            if field_name in ["bottle_id", "images", "error_message"]:
                continue  # Skip metadata fields
            corrections.append(
                Correction(
                    field_name=field_name,
                    original_value="",
                    corrected_value=str(correction_value),
                )
            )

    # Check hints.md for additional guidance
    hints_file = args.report.parent.parent / "hints.md"
    hints_text = ""

    if hints_file.exists():
        try:
            with hints_file.open("r", encoding="utf-8") as f:
                hints_text = f.read()
            # Extract hints relevant to this bottle
            import re

            bottle_hints_pattern = re.compile(
                rf".*{re.escape(bottle_id)}.*?:(.*?)(?=\n\n|\n[0-9]+\.|\Z)", re.MULTILINE | re.DOTALL
            )
            matches = bottle_hints_pattern.findall(hints_text)
            if matches:
                hints_text = "\n".join(matches).strip()
                logger.info("Found hints for bottle %s", bottle_id)
        except Exception:
            logger.warning("Failed to read hints.md")

    # Re-process the bottle
    try:
        # Apply hints as corrections if applicable
        if hints_text and not corrections:
            logger.info("Applying hints from hints.md")
            # Hints could be applied as context to the analyzer
            # For now, just log them as they may need manual interpretation

        bottle_dict, metadata = bottle_analyzer.process_single_bottle(
            [images],
            vision_model="qwen2-vl:7b",
        )

        # Apply suggested corrections
        if corrections:
            batch_processor = BatchProcessor()
            bottle_dict = batch_processor.apply_corrections([bottle_dict], corrections)[0]
            logger.info("Applied %d corrections from suggested_fix", len(corrections))

        # Create Bottle object
        bottle = Bottle(
            name=bottle_dict.get("name", "Unknown"),
            distillery=bottle_dict.get("distillery", "Unknown"),
            msrp=float(bottle_dict.get("msrp", "0.00")),
            specific_release=bottle_dict.get("specific_release", ""),
            proof=bottle_dict.get("proof"),
            age=bottle_dict.get("age"),
            tasting_notes=bottle_dict.get("tasting_notes"),
            metadata=metadata,
            corrections=corrections,
        )

        # Save output files
        from image_processor.preview_generator import PreviewGenerator

        preview_generator = PreviewGenerator(output_dir=args.output_dir)
        output_folder = preview_generator.save_bottle_files(bottle, metadata)
        logger.info("Successfully reprocessed bottle, saved to: %s", output_folder)

        return 0

    except Exception:
        logger.exception("Failed to reprocess bottle")
        return 1


def main(argv: list | None = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Optional list of arguments (default: sys.argv[1:]).

    Returns:
        Exit code.
    """
    args = parse_args(argv)

    # Setup logging
    setup_logging(debug=args.debug)

    # Default to process command if no command specified
    if not args.command or args.command == "process":
        return process_command(args)
    elif args.command == "feedback":
        return feedback_command(args)
    elif args.command == "retry":
        return retry_command(args)
    else:
        logger.error("Unknown command: %s", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
