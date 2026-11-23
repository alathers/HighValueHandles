"""Generate preview reports and output files for bourbon bottles.

Creates markdown output, saves images, generates error reports,
and provides preview for user review.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from lib.error_handling import ErrorType, ErrorReport, create_error_report, save_error_reports
from lib.exceptions import (
    BourbonException,
    OCRError,
    AIModelError,
    ImageProcessingError,
    ConfigurationError,
    FileSystemError,
    ValidationError,
)
from models.bottle import Bottle, ExtractedMetadata

logger = logging.getLogger(__name__)


class PreviewGenerator:
    """Generate preview reports and output files for bourbon bottles.

    Handles markdown generation, file saving, and error report creation.

    Attributes:
        output_dir: Base output directory for processed bottles.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize preview generator.

        Args:
            output_dir: Base output directory (default: "processed").
        """
        self.output_dir = Path(output_dir) if output_dir else Path("processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_bottle_markdown(self, bottle: Bottle) -> str:
        """Generate markdown content for bottle.

        Args:
            bottle: Bottle model instance.

        Returns:
            Markdown formatted string.
        """
        lines = [f"# {bottle.name}\n", "\n"]

        # Distillery
        lines.append("## Distillery")
        lines.append(f"{bottle.distillery}\n")

        # MSRP
        lines.append("\n## MSRP")
        lines.append(f"${bottle.msrp}\n")

        # Release Information section
        lines.append("\n## Release Information")

        if bottle.proof:
            lines.append(f"- **Proof:** {bottle.proof}")

        if bottle.age:
            lines.append(f"- **Age:** {bottle.age}")

        if bottle.specific_release:
            lines.append(f"- **Release:** {bottle.specific_release}")

        # Tasting Notes
        if bottle.tasting_notes:
            lines.append("\n## Tasting Notes")
            lines.append(bottle.tasting_notes)

        # Reference Links
        if bottle.reference_links:
            lines.append("\n## References")
            for ref in bottle.reference_links:
                lines.append(f"[{ref.label}]({ref.url})")

        # Quantity section
        lines.append("\n## Quantity")
        lines.append(f"Total quantity: {bottle.quantity.count}")
        if bottle.quantity.count > 1:
            unit_value = bottle.total_value / bottle.quantity.count
            lines.append(f"Value per bottle: ${unit_value:.2f}")
        lines.append(f"Total value: ${bottle.total_value:.2f}")

        # Images
        lines.append("\n## Images")
        for i, img in enumerate(bottle.metadata.source_images, 1):
            img_name = Path(img).name
            lines.append(f"{i}. {img_name}")

        # Processing Notes
        if bottle.corrections:
            lines.append("\n## Corrections")
            for correction in bottle.corrections:
                lines.append(f"- **{correction.field_name}**: `{correction.original_value}` → `{correction.corrected_value}`")

        return "\n".join(lines)

    def save_bottle_files(
        self,
        bottle: Bottle,
        metadata: ExtractedMetadata,
        prefix: Optional[str] = None,
    ) -> Path:
        """Save output files for processed bottle.

        Creates sub-folder, copies images, writes markdown and OCR text.

        Args:
            bottle: Bottle model instance.
            metadata: ExtractedMetadata from analysis.
            prefix: Optional prefix for folder name (default: auto-generated from name).

        Returns:
            Path to created output folder.
        """
        # Generate folder name
        if prefix:
            folder_name = prefix
        else:
            # Use bottle name, sanitize for filesystem
            safe_name = bottle.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{safe_name}_{timestamp}"

        output_folder = self.output_dir / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_path in bottle.metadata.source_images:
            src = Path(img_path)
            if src.exists():
                dst = output_folder / src.name
                import shutil

                shutil.copy2(src, dst)
                logger.debug(f"Copied image to {dst}")

        # Write markdown
        markdown_content = self.generate_bottle_markdown(bottle)
        markdown_path = output_folder / "bottle.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Wrote markdown to {markdown_path}")

        # Write OCR text
        ocr_path = output_folder / "extracted_text.txt"
        with open(ocr_path, "w", encoding="utf-8") as f:
            f.write(metadata.full_text or "")
        logger.info(f"Wrote OCR text to {ocr_path}")

        return output_folder

    def generate_error_report(
        self,
        bottle_id: str,
        images: List[str],
        error: Exception,
        retry_attempts: int = 0,
    ) -> ErrorReport:
        """Generate error report for failed processing.

        Args:
            bottle_id: Bottle identifier.
            images: List of image paths that failed.
            error: Exception that occurred.
            retry_attempts: Number of retry attempts made.

        Returns:
            ErrorReport instance.
        """
        # Determine error type and suggested fix
        if isinstance(error, OCRError):
            suggested_fix = "Check image quality and try re-scanning with higher resolution."
            hint_topic = ErrorType.OCR_FAILURE
        elif isinstance(error, AIModelError):
            suggested_fix = "Verify Ollama service is running and model is available."
            hint_topic = ErrorType.AI_MODEL_FAILURE
        elif isinstance(error, (ImageProcessingError, ConfigurationError, FileSystemError, ValidationError)):
            suggested_fix = "Review error details and check system configuration."
            hint_topic = ErrorType.UNKNOWN
        elif isinstance(error, BourbonException):
            error_msg = str(error).lower()
            if "model" in error_msg or "ollama" in error_msg:
                suggested_fix = "Verify Ollama service is running and model is available."
                hint_topic = ErrorType.AI_MODEL_FAILURE
            elif "ocr" in error_msg or "tesseract" in error_msg:
                suggested_fix = "Check image quality and try re-scanning with higher resolution."
                hint_topic = ErrorType.OCR_FAILURE
            else:
                suggested_fix = "Review error details and check system configuration."
                hint_topic = ErrorType.UNKNOWN
        else:
            suggested_fix = "Check system logs and verify all dependencies are installed."
            hint_topic = ErrorType.UNKNOWN

        return ErrorReport(
            error_type=hint_topic,
            file_path=Path(images[0]) if images else None,
            message=str(error),
            details=f"Bottle ID: {bottle_id}\nImages: {images}",
            timestamp=datetime.utcnow().isoformat(),
            retry_count=retry_attempts,
            hint=suggested_fix,
        )

    def save_error_report(
        self,
        error_report: ErrorReport,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save error report to file.

        Args:
            error_report: ErrorReport instance.
            output_dir: Optional output directory (default: output_dir).

        Returns:
            Path to saved error report file.
        """
        base_dir = output_dir or self.output_dir

        # Generate filename from timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = base_dir / f"error_report_{timestamp}.json"

        # Save error report
        save_error_reports([error_report], report_path)

        logger.info(f"Saved error report to {report_path}")
        return report_path

    def generate_batch_preview_report(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        rejected_bottles: Optional[List[str]] = None,
        batch_id: int = 1,
    ) -> str:
        """Generate preview report for multiple bottles with grouping summary table.

        Creates markdown with grouping summary table (Bottle Name, Quantity, Images,
        Confidence), extracted metadata details per bottle, rejected/low-confidence items,
        and instructions for feedback.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.
            rejected_bottles: List of rejected bottle identifiers.
            batch_id: Batch identifier for this report.

        Returns:
            Markdown formatted preview report.
        """
        lines = [f"# Batch Preview - Batch {batch_id}\n", "\n"]
        lines.append(f"Total bottles in batch: {len(bottle_dicts)}\n")

        if rejected_bottles:
            lines.append(f"Rejected bottles: {len(rejected_bottles)}\n")

        lines.append("\n## Grouping Summary\n")
        lines.append("| Bottle Name | Quantity | Images | Confidence |\n")
        lines.append("|-------------|----------|--------|------------|\n")

        for bottle_dict, metadata in zip(bottle_dicts, metadata_list):
            name = bottle_dict.get("name", "Unknown")
            quantity = bottle_dict.get("quantity", 1)
            image_count = len(metadata.source_images)

            # Calculate confidence level
            filled_fields = sum(
                1
                for field in ["name", "distillery", "msrp", "proof", "age"]
                if bottle_dict.get(field)
            )
            if filled_fields >= 4:
                confidence = "HIGH"
            elif filled_fields >= 2:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            lines.append(f"| {name} | {quantity} | {image_count} | {confidence} |\n")

        # Rejected/low-confidence items section
        low_confidence_bottles = []
        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list)):
            filled_fields = sum(
                1
                for field in ["name", "distillery", "msrp", "proof", "age"]
                if bottle_dict.get(field)
            )
            if filled_fields < 2:
                low_confidence_bottles.append(i + 1)

        if low_confidence_bottles or rejected_bottles:
            lines.append("\n## Low-Confidence or Rejected Items\n")

            if low_confidence_bottles:
                lines.append("**Low Confidence:**\n")
                for idx in low_confidence_bottles:
                    lines.append(f"- Bottle {idx} (check extracted metadata)\n")

            if rejected_bottles:
                lines.append("\n**Rejected:**\n")
                for bottle_id in rejected_bottles:
                    lines.append(f"- {bottle_id}\n")

        # Detailed metadata per bottle
        lines.append("\n## Extracted Metadata Details\n\n")

        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list), 1):
            lines.append(f"### Bottle {i}\n")
            lines.append(f"**Name:** {bottle_dict.get('name', 'Unknown')}\n")
            lines.append(f"**Distillery:** {bottle_dict.get('distillery', 'Unknown')}\n")
            lines.append(f"**MSRP:** ${bottle_dict.get('msrp', '0.00')}\n")

            if bottle_dict.get("proof"):
                lines.append(f"**Proof:** {bottle_dict['proof']}\n")

            if bottle_dict.get("age"):
                lines.append(f"**Age:** {bottle_dict['age']}\n")

            if bottle_dict.get("specific_release"):
                lines.append(f"**Release:** {bottle_dict['specific_release']}\n")

            lines.append(f"**Quantity:** {bottle_dict.get('quantity', 1)}\n")

            # Images section
            lines.append(f"**Images:** {len(metadata.source_images)}\n")
            for img_path in metadata.source_images:
                img_name = Path(img_path).name
                lines.append(f"- {img_name}\n")

            # OCR preview
            if metadata.full_text:
                lines.append("\n**OCR Preview:**\n")
                lines.append("```\n")
                preview = metadata.full_text[:300]
                if len(metadata.full_text) > 300:
                    preview += "..."
                lines.append(f"{preview}\n")
                lines.append("```\n")

            lines.append("\n")

        # Feedback instructions
        lines.append("## Feedback Instructions\n")
        lines.append("Review the batch grouping and metadata above. To provide feedback:\n\n")
        lines.append(
            "1. **Interactive Mode:** Use the [A]ccept/[R]eject/[M]erge/[S]plit/[E]dit options\n"
        )
        lines.append("   - **A** - Accept the grouping and metadata as-is\n")
        lines.append("   - **R** - Reject this bottle for re-processing\n")
        lines.append("   - **M** - Merge multiple bottles into one\n")
        lines.append("   - **S** - Split a single proposed bottle into multiple bottles\n")
        lines.append("   - **E** - Edit specific fields (name, distillery, MSRP, etc.)\n\n")
        lines.append(
            "2. **Separate Feedback Command:** Use the `feedback` command with a JSON corrections file\n"
        )
        lines.append("   ```\n")
        lines.append("   python -m bourbon_cli feedback --batch 1 --corrections feedback.json\n")
        lines.append("   ```\n\n")
        lines.append(
            "3. **Retry Failed Items:** Use the `retry` command with an error report file\n"
        )
        lines.append("   ```\n")
        lines.append("   python -m bourbon_cli retry --report error_report.json\n")
        lines.append("   ```\n")

        return "\n".join(lines)

    def save_batch_preview(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        batch_id: int = 1,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save batch preview report to file with batch-specific naming.

        Writes markdown preview to processed/batch_{N}_preview.md with per-batch
        naming to support multi-batch processing without file conflicts.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.
            batch_id: Batch identifier for filename generation.
            output_path: Optional output path (default: output_dir/batch_{batch_id}_preview.md).

        Returns:
            Path to saved preview report.
        """
        if output_path is None:
            output_path = self.output_dir / f"batch_{batch_id}_preview.md"

        markdown = self.generate_batch_preview_report(
            bottle_dicts,
            metadata_list,
            batch_id=batch_id,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info(f"Saved batch preview to {output_path}")
        return output_path

    def save_batch_preview_report(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save batch preview report to file (legacy method, uses default name).

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata
            output_path: Optional output path (default: output_dir/batch_preview.md).

        Returns:
            Path to saved preview report.
        """
        logger.warning(
            "save_batch_preview_report() is deprecated, use save_batch_preview() with batch_id for batch-specific naming"
        )
        return self.save_batch_preview(bottle_dicts, metadata_list, batch_id=1, output_path=output_path)

    def interactive_prompt(
        self,
        bottle_dicts: List[Dict],
        metadata_list: List[ExtractedMetadata],
    ) -> tuple[List[Dict[str, str]], List[str]]:
        """Prompt user interactively for corrections after batch processing.

        Displays each bottle's key information and waits for user input.
        Supports [A]ccept, [R]eject, [M]erge, [S]plit, [E]dit, and [Q]uit actions.

        Args:
            bottle_dicts: List of bottle dictionaries.
            metadata_list: List of ExtractedMetadata.

        Returns:
            Tuple of (corrections, rejected_bottles) where:
                - corrections: List of correction dicts (field_name, original_value, corrected_value)
                - rejected_bottles: List of bottle identifiers marked for rejection
        """
        from datetime import datetime

        corrections = []
        rejected_bottles = []

        print("\n" + "=" * 70)
        print("BATCH PROCESSING - INTERACTIVE REVIEW")
        print("=" * 70)
        print(f"\nProcessing {len(bottle_dicts)} bottles in this batch")
        print("Actions: [A]ccept | [R]eject | [M]erge | [S]plit | [E]dit | [Q]uit\n")

        for i, (bottle_dict, metadata) in enumerate(zip(bottle_dicts, metadata_list), 1):
            # Display bottle identifier
            bottle_id = Path(metadata.source_images[0]).stem if metadata.source_images else f"bottle_{i}"
            print("-" * 70)
            print(f"Bottle {i}/{len(bottle_dicts)} - ID: {bottle_id}")
            print("-" * 70)

            # Display key bottle information
            print(f"\nName:        {bottle_dict.get('name', 'Unknown')}")
            print(f"Distillery:  {bottle_dict.get('distillery', 'Unknown')}")
            print(f"MSRP:        ${bottle_dict.get('msrp', '0.00')}")

            # Display optional fields if present
            proof = bottle_dict.get("proof")
            if proof:
                print(f"Proof:       {proof}")

            age = bottle_dict.get("age")
            if age:
                print(f"Age:         {age}")

            release = bottle_dict.get("specific_release")
            if release:
                print(f"Release:     {release}")

            quantity = bottle_dict.get("quantity", 1)
            print(f"Quantity:    {quantity}")

            # Display image count
            image_count = len(metadata.source_images)
            print(f"Images:      {image_count}")

            # Check for low confidence
            key_fields = ["name", "distillery", "msrp", "proof", "age"]
            filled_fields = sum(1 for field in key_fields if bottle_dict.get(field))
            confidence = "HIGH" if filled_fields >= 4 else "MEDIUM" if filled_fields >= 2 else "LOW"
            print(f"Confidence:  {confidence}")

            # Wait for user input
            print("\nAction [A/R/M/S/E/Q]? ", end="", flush=True)
            action = input().strip().upper()

            # Handle user action
            if action == "Q":
                print("\nQuitting interactive review...")
                break
            elif action == "A":
                print(f"✓ Accepted bottle {i}: {bottle_dict.get('name', 'Unknown')}")
            elif action == "R":
                rejected_bottles.append(bottle_id)
                print(f"✗ Rejected bottle {i}: {bottle_id}")
            elif action == "M":
                # Merge with previous bottle
                if i > 1:
                    prev_id = i - 1
                    print(f"→ Merging bottle {i} with bottle {prev_id}")
                    corrections.append(
                        {
                            "field_name": "grouping",
                            "original_value": f"separate_{i}",
                            "corrected_value": f"merge_with_{prev_id}",
                            "bottle_id": bottle_id,
                        }
                    )
                else:
                    print(f"⚠ Cannot merge first bottle (no previous bottle to merge with)")
            elif action == "S":
                # Split this bottle (indicates grouping is too aggressive)
                print(f"→ Splitting bottle {i} into separate items")
                corrections.append(
                    {
                        "field_name": "grouping",
                        "original_value": f"grouped_{i}",
                        "corrected_value": "split",
                        "bottle_id": bottle_id,
                    }
                )
            elif action == "E":
                # Edit specific fields
                print(f"\nEditing bottle {i}: {bottle_dict.get('name', 'Unknown')}")
                print("Press Enter to keep current value for each field\n")

                for field in ["name", "distillery", "msrp", "proof", "age", "specific_release"]:
                    current = bottle_dict.get(field, "")
                    label = field.replace("_", " ").title()
                    new_value = input(f"{label:14} [{current}]: ").strip()

                    if new_value and new_value != current:
                        corrections.append(
                            {
                                "field_name": field,
                                "original_value": str(current),
                                "corrected_value": new_value,
                                "bottle_id": bottle_id,
                                "corrected_at": datetime.utcnow().isoformat(),
                            }
                        )
                        # Update bottle dict for display
                        bottle_dict[field] = new_value
                        print(f"  Updated: {field} = {new_value}")
            else:
                print(f"⚠ Unknown action '{action}', treating as [A]ccept")

        # Summary
        print("\n" + "=" * 70)
        print("INTERACTIVE REVIEW SUMMARY")
        print("=" * 70)
        accepted = len(bottle_dicts) - len(rejected_bottles)
        print(f"Accepted:     {accepted}/{len(bottle_dicts)}")
        print(f"Rejected:     {len(rejected_bottles)}")
        print(f"Corrections:  {len(corrections)}")
        print("=" * 70)

        return corrections, rejected_bottles

    def handle_interactive_choice(
        self,
        choice: str,
        bottle_dict: Dict,
        bottle_id: str,
        bottle_index: int,
    ) -> Optional[Dict[str, str]]:
        """Parse user interactive choice and return appropriate action.

        Handles A/R/M/S/E choices and returns a correction dict for actions
        that require processing (Reject, Merge, Split, Edit).

        Args:
            choice: User input choice (A/R/M/S/E/Q).
            bottle_dict: Bottle dictionary for this bottle.
            bottle_id: Bottle identifier.
            bottle_index: Index within batch (1-based).

        Returns:
            Correction dict if action requires processing, None otherwise.
            Returns None for Accept actions (no correction needed).
        """
        action = choice.strip().upper()

        if action == "A":
            # Accept - no correction needed
            return None
        elif action == "R":
            # Reject - mark for deletion
            return {
                "field_name": "grouping",
                "original_value": "accepted",
                "corrected_value": "rejected",
                "bottle_id": bottle_id,
                "corrected_at": datetime.utcnow().isoformat(),
            }
        elif action == "M":
            # Merge with previous bottle
            if bottle_index > 1:
                prev_id = bottle_index - 1
                return {
                    "field_name": "grouping",
                    "original_value": f"separate_{bottle_index}",
                    "corrected_value": f"merge_with_{prev_id}",
                    "bottle_id": bottle_id,
                    "corrected_at": datetime.utcnow().isoformat(),
                }
            else:
                logger.warning("Cannot merge first bottle (no previous bottle to merge with)")
                return None
        elif action == "S":
            # Split this bottle
            return {
                "field_name": "grouping",
                "original_value": f"grouped_{bottle_index}",
                "corrected_value": "split",
                "bottle_id": bottle_id,
                "corrected_at": datetime.utcnow().isoformat(),
            }
        elif action == "E":
            # Edit - requires additional context (field-specific)
            # This returns a marker that indicates edit mode should be entered
            return {
                "field_name": "edit",
                "original_value": "",
                "corrected_value": "enter_edit_mode",
                "bottle_id": bottle_id,
                "corrected_at": datetime.utcnow().isoformat(),
            }
        else:
            logger.warning("Unknown choice '%s', treating as Accept", choice)
            return None
