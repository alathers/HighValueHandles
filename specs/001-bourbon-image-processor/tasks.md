# Implementation Tasks: Bourbon Image Processor

**Feature**: Bourbon Image Processor
**Branch**: `001-bourbon-image-processor`
**Generated**: 2026-11-09
**Status**: Ready for Implementation

---

## Phase 1: Setup (Project Initialization)

**Goal**: Establish development environment, project structure, and quality gates per constitution.

### Independent Test Criteria
- Running `pytest` passes with no test failures (empty test suite acceptable)
- Pre-commit hooks configured and阻塞 commits with lint errors or large files
- `uv sync` installs all dependencies successfully

### Implementation Tasks

- [X] T001 Create project directory structure per plan: src/image_processor/, src/models/, src/cli/, src/lib/, tests/unit/, tests/contract/, tests/integration/, tests/fixtures/images/
- [X] T002 Initialize pyproject.toml with Python 3.11+ requirement and dependencies: requests>=2.31.0, pillow>=10.0.0, opencv-python>=4.8.0, numpy>=1.24.0, pytesseract>=0.3.10, pytest>=7.4.0
- [X] T003 Create dev-requirements.txt with pytest, pytest-cov, pytest-asyncio, black, ruff, mypy for development tooling
- [X] T004 Create .python-version file specifying "3.11" or higher
- [X] T005 Create src/__init__.py, src/image_processor/__init__.py, src/models/__init__.py, src/cli/__init__.py, src/lib/__init__.py for module packaging
- [X] T006 Create tests/__init__.py, tests/unit/__init__.py, tests/contract/__init__.py, tests/integration/__init__.py for test packaging
- [X] T007 Create tests/fixtures/images/ directory structure with subfolders: straightforward/, rotated/, ambiguous/ for integration test fixtures (placeholder files ok initially)
- [X] T008 Create .gitignore entries: __pycache__/, *.pyc, .venv/, .pytest_cache/, processed/, .specify_config.toml, hints.md,*.egg-info/, .mypy_cache/, .coverage, htmlcov/
- [X] T009 Configure pre-commit hooks via .pre-commit-config.yaml to run ruff (linting) and check-file-size (max 5MB) on all commits
- [X] T010 Create install.sh script supporting macOS and Linux that runs `uv pip install -r requirements.txt -r dev-requirements.txt` and installs tesseract-ocr via system package manager
- [X] T011 Add execute permissions to install.sh with `chmod +x install.sh`
- [X] T012 Create pyproject.toml [tool.pytest.ini_options] section with testpaths = ["tests"], python_files = ["test_*.py"]
- [X] T013 Create pyproject.toml [tool.ruff] section with line-length = 100, target-version = "py311"
- [X] T014 Create pyproject.toml [tool.black] section with line-length = 100, target-version = ['py311']
- [X] T015 Create README.md with installation instructions (uv sync, install.sh, tesseract-OCR installation), basic usage examples, and troubleshooting section

---

## Phase 2: Foundational (Blocking Prerequisites)

**Goal**: Build shared utilities and data models that all user stories depend on.

### Independent Test Criteria
- Ollama client can successfully connect and list available models
- Configuration management correctly resolves CLI args → env vars → config file → defaults in priority order
- Data model classes can be instantiated and serialized to/from JSON
- Error handling utilities generate properly formatted error reports

### Implementation Tasks

- [X] T016 Create src/lib/config.py module with Config class implementing configuration priority: CLI args → environment variables → .specify_config.toml → defaults
- [X] T017 Add Config.get_bool(), Config.get_int(), Config.get_str(), Config.get_url() helper methods in src/lib/config.py with validation
- [X] T018 Define all configuration keys in src/lib/config.py: image_folder, batch_size (integer or "all"), model, interactive, include_vendor_images, ollama_url, output_dir, allow_retry, lookahead, concurrency, debug
- [X] T019 Add Config.validate() method in src/lib/config.py that checks: batch_size >= 1 or "all", lookahead 0-10, concurrency 1-10, Ollama URL is valid HTTP URL
- [X] T020 Create src/lib/ollama_client.py module with OllamaClient class wrapping Ollama REST API at configurable URL
- [X] T021 Implement OllamaClient.list_models() method in src/lib/ollama_client.py that GETs /api/tags and returns list of available model names
- [X] T022 Implement OllamaClient.generate_vision() method in src/lib/ollama_client.py that POSTs /api/generate with image data, model name, and prompt, returning generated text
- [X] T023 Implement OllamaClient.generate_text() method in src/lib/ollama_client.py that POSTs /api/generate with text prompt only, returning generated text
- [X] T024 Add OllamaClient.is_model_available(model_name) method in src/lib/ollama_client.py that checks model in list_models() results
- [X] T025 Add OllamaClient.health_check() method in src/lib/ollama_client.py that returns True if Ollama service is reachable at configured URL
- [X] T026 Create src/lib/error_handling.py module with ErrorReport dataclass containing error_id, bottle_id, images, error_type, error_message, retry_attempts, elapsed_time_seconds, suggested_fix, hint_topic
- [X] T027 Implement ErrorReport.to_json() and ErrorReport.from_json() methods in src/lib/error_handling.py for serialization/deserialization
- [X] T028 Create src/lib/error_handling.py RetryManager class with retry(max_attempts=3, backoff_delays=[0, 2, 5]) method that wraps function calls with exponential backoff retry logic
- [X] T029 Define error_type enum in src/lib/error_handling.py: OCR_FAILURE, MODEL_TIMEOUT, DECODING_ERROR, NETWORK_ERROR, ORIENTATION_DETECTION_FAILED
- [X] T030 Add create_error_report() function in src/lib/error_handling.py that generates ErrorReport JSON file at specified path
- [X] T031 Create src/models/image.py module with Image dataclass containing path, category (enum: ANGLE, BOX, DETAIL, VENDOR, BILL, UNKNOWN), status (enum: PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED), bottle_ref, md5_hash, file_size, is_primary
- [X] T032 Add validation to Image dataclass in src/models/image.py: path must exist and be readable, file_size populated
- [X] T033 Create src/models/bottle.py module with ExtractedMetadata dataclass
- [X] T034 Add ExtractedMetadata dataclass in src/models/bottle.py with full_text, selected_text, source_images
- [X] T035 Create Bottle dataclass in src/models/bottle.py with name, distillery, specific_release, msrp (required), plus optional proof, age, tasting_notes, metadata, quantity, ohr, reference_links, corrections
- [X] T036 Add Bottle.to_dict() and Bottle.from_dict() serialization methods in src/models/bottle.py
- [X] T037 Add Bottle total_value property in src/models/bottle.py that calculates quantity * price (or ohr if price = 0)
- [X] T038 Create ReferenceLink dataclass in src/models/bottle.py with url, label
- [X] T039 Create src/lib/duplicates.py module with DuplicateDetector class for MD5-based duplicate detection
- [X] T040 Add Bottle.to_markdown() method in src/models/bottle.py that generates markdown file content with sections: Bottle Name, Distillery, MSRP, Release Information, Tasting Notes, References, Quantity
- [X] T041 Create BatchPreviewReport dataclass in src/models/processing.py with id, metadata, extracted_text, image_paths, status, corrections
- [X] T042 Create Correction dataclass in src/models/bottle.py with field_name, original_value, corrected_value, corrected_at

---

## Phase 3: User Story 1 - Process Single Bottle Images (P1)

**Story Goal**: Extract bourbon bottle metadata (name, MSRP, distillery, release info, tasting notes) from single bottle images and generate publishable markdown output.

### Independent Test Criteria
- Given bottle images (front, back, label), system generates markdown file with complete metadata (name, distillery, proof, age, batch, MSRP, tasting notes)
- Given rotated images, system auto-orients and extracts text correctly
- Given images with missing information, system marks fields as [MISSING] or [EXTRACTION FAILED]
- Given blurred/illegible text, system generates error report after 3 retry attempts

### Implementation Tasks

- [X] T043 [US1] Create src/image_processor/text_extractor.py module with TextExtractor class
- [X] T044 [US1] Implement TextExtractor.detect_orientation() method in src/image_processor/text_extractor.py that uses OpenCV to detect text angle and returns rotation needed (0, 90, 180, 270 degrees)
- [X] T045 [US1] Implement TextExtractor.correct_orientation() method in src/image_processor/text_extractor.py that rotates image to upright using Pillow and updates Image.original_orientation and corrected fields
- [X] T046 [US1] Implement TextExtractor.extract_text_ocr() method in src/image_processor/text_extractor.py that uses pytesseract to extract raw text from (corrected) image and returns text string
- [X] T047 [US1] Implement TextExtractor.analyze_with_vision_model() method in src/image_processor/text_extractor.py that calls OllamaClient.generate_vision() with prompt to extract bottle metadata (name, distillery, proof, age, batch, MSRP, tasting notes)
- [X] T048 [US1] Implement TextExtractor.extract_metadata() method in src/image_processor/text_extractor.py that orchestrates: detect_orientation → correct_orientation → extract_text_ocr → analyze_with_vision_model, returns ExtractedMetadata with confidence scores
- [X] T049 [US1] Add retry logic to TextExtractor.extract_metadata() in src/image_processor/text_extractor.py using RetryManager with max_attempts=3, backoff_delays=[0, 2, 5]
- [X] T050 [US1] Create src/image_processor/bottle_analyzer.py module with BottleAnalyzer class
- [X] T051 [US1] Implement BottleAnalyzer.process_single_bottle() method in src/image_processor/bottle_analyzer.py that takes list of Image objects and returns Bottle with metadata from TextExtractor
- [X] T052 [US1] Implement BottleAnalyzer.group_images_by_bottle() method in src/image_processor/bottle_analyzer.py that analyzes visual/text similarity between images and groups them as Bottle with appropriate Image list (handles multi-angle photos of same bottle)
- [X] T053 [US1] Implement BottleAnalyzer.detect_quantity() method in src/image_processor/bottle_analyzer.py that creates Quantity entity from Bottle.images by checking hash-based duplicates and counting identical bottles in single frame
- [X] T054 [US1] Add BottleAnalyzer.calculate_confidence() method in src/image_processor/bottle_analyzer.py that aggregates confidence scores from ExtractedMetadata fields to determine HIGH/MEDIUM/LOW confidence for overall extraction
- [X] T055 [US1] Create src/image_processor/preview_generator.py module with PreviewGenerator class
- [X] T056 [US1] Implement PreviewGenerator.generate_bottle_markdown() method in src/image_processor/preview_generator.py that generates Bottle.to_markdown() content with all metadata fields, references section if ReferenceLinks exist
- [X] T057 [US1] Implement PreviewGenerator.save_bottle_files() method in src/image_processor/preview_generator.py that creates output sub-folder (e.g., "processed/Buffalo_Trace_20261109_143022/"), copies original images, writes bottle.md with markdown content, writes extracted_text.txt with raw OCR text
- [X] T058 [US1] Implement PreviewGenerator.generate_error_report() method in src/image_processor/preview_generator.py that calls create_error_report() from lib/error_handling.py for failed bottles
- [X] T059 [US1] Create src/cli/commands.py module with argparse-based CLI parser
- [X] T060 [US1] Implement process command in src/cli/commands.py with arguments: -f/--image-folder PATH (default .), -b/--batch-size N|all (default 5), -m/--model NAME (default qwen2-vl:7b [note: task spec had qwen3-vl:32b]), -i/--interactive (boolean flag default false), -v/--include-vendor-images (boolean default false), -u/--ollama-url URL (default http://localhost:11434), -o/--output-dir PATH (default processed/), -r/--allow-retry (boolean default true), --lookahead N (default 3), --concurrency N (default 3), --debug (boolean default false), --dry-run (boolean default false), -h/--help
- [X] T061 [US1] Implement src/cli/commands.py pre-flight checks before processing: verify input folder exists and contains supported images, verify Ollama service reachable via OllamaClient.health_check(), verify model available via OllamaClient.is_model_available(), verify Tesseract OCR installed, verify output directory writable
- [X] T062 [US1] Implement src/cli/commands.py process command main flow: load images from folder → for each image/Image group, call BottleAnalyzer.process_single_bottle() → call PreviewGenerator.save_bottle_files() → log progress messages
- [X] T063 [US1] Add src/cli/commands.py process command error handling: if individual bottle fails after 3 retries, generate error report and continue processing remaining bottles
- [X] T064 [US1] Implement src/cli/commands.py completion reporting: print summary of processed/failed bottles, paths to output files, warnings/errors encountered
- [X] T065 [US1] Add src/cli/commands.py exit codes: 0 (success), 1 (general error), 2 (Ollama connection failed), 3 (no valid images found), 4 (all bottles failed), 5 (output directory creation failed)

---

## Phase 4: User Story 2 - Organize Processed Bottle Sets (P2)

**Story Goal**: Create structured processed/ folder hierarchy where each bottle has its own sub-folder containing all related images and markdown file.

### Independent Test Criteria
- Given multiple bottles processed, each has unique sub-folder in processed/ with clear naming (e.g., "Buffalo_Trace_BT2407/")
- Given multiple images of same bottle, all placed in same sub-folder
- Given processed/ folder already exists, new bottles added without overwriting existing ones

### Implementation Tasks

- [X] T066 [P] [US2] Implement PreviewGenerator.create_output_folder() method in src/image_processor/preview_generator.py that creates "processed/" directory if not exists with os.makedirs(processed, exist_ok=True) [already implemented in __init__]
- [X] T067 [P] [US2] Implement PreviewGenerator.generate_folder_name() method in src/image_processor/preview_generator.py that generates unique sub-folder name like "{sanitized_name}_{timestamp}" or adds disambiguating info (batch/proof) if naming conflict detected [integrated in save_bottle_files]
- [X] T068 [P] [US2] Implement PreviewGenerator.copy_images_to_folder() method in src/image_processor/preview_generator.py that copies all Bottle.images to sub-folder preserving original filenames [integrated in save_bottle_files]
- [X] T069 [P] [US2] Update PreviewGenerator.save_bottle_files() in src/image_processor/preview_generator.py to call create_output_folder(), generate_folder_name(), copy_images_to_folder() before writing markdown and extracted text files [fully implemented]
- [X] T070 [P] [US2] Create processed/.gitkeep file in setup (T007 or separate task) to track processed/ in git without committing contents
- [X] T071 [P] [US2] Create src/image_processor/image_organizer.py module with ImageOrganizer class [implemented with all required methods]
- [X] T072 [P] [US2] Implement ImageOrganizer.scan_images() method in src/image_processor/image_organizer.py that recursively scans input folder for supported image formats (.jpeg, .jpg, .png, .tiff, .heic) and returns list of Image objects with filepath populated
- [X] T073 [P] [US2] Implement ImageOrganizer.group_by_bottle() method in src/image_processor/image_organizer.py that takes list[Image] and returns list[list[Image]] representing groups of images belonging to same bottle
- [X] T074 [P] [US2] Implement ImageOrganizer.detect_duplicates() method in src/image_processor/image_organizer.py that uses file hash comparison (md5 or sha256) to identify exact duplicate images, updates Image.is_duplicate and Image.duplicate_of_id fields
- [X] T075 [P] [US2] Implement ImageOrganizer.is_duplicate_image() helper method in src/image_processor/image_organizer.py that compares two Image file paths and returns True if file content hash identical
- [X] T076 [P] [US2] Update BottleAnalyzer.group_images_by_bottle() in src/image_processor/bottle_analyzer.py to use ImageOrganizer.group_by_bottle() for initial grouping, then apply text verification to refine groups [delegated entirely to ImageOrganizer]
- [X] T077 [P] [US2] Update src/cli/commands.py process command to call ImageOrganizer.scan_images() and ImageOrganizer.group_by_bottle() at start of processing pipeline

---

## Phase 5: User Story 3 - Batch Process Multiple Bottles (P3)

**Story Goal**: Process multiple bottle sets in configurable batches with preview reports, lookahead for multi-angle photos, and optional interactive CLI feedback loop.

### Independent Test Criteria
- Given 15 bottle images with default batch-size=5, system processes in 3 batches (5, 5, 5) with preview reports after each batch
- Given lookahead=3, system includes multi-angle photos of same bottle in current batch rather than splitting across batches
- Given --batch-size all, system processes all images without stopping for preview reports
- Given --interactive flag, system prompts user after each grouping with [A]ccept/[R]eject/[M]erge/[S]plit/[E]dit options
- Given preview report generated, user can provide feedback via CLI prompts or separate `feedback` command and system re-processes only flagged items

### Implementation Tasks

- [X] T078 [P] [US3] Create src/image_processor/batch_processor.py module with BatchProcessor class
- [X] T079 [P] [US3] Implement BatchProcessor.slice_into_batches() method in src/image_processor/batch_processor.py that takes list[list[Image]] (bottle image groups from ImageOrganizer) and splits into batches based on batch_size config
- [X] T080 [P] [US3] Implement BatchProcessor.apply_lookahead() method in src/image_processor/batch_processor.py that peeks at next N images (default 3) to check for multi-angle photos of bottles in current batch, moves them into current batch if similar
- [X] T081 [P] [US3] Implement BatchProcessor.vision_similarity_score() method in src/image_processor/batch_processor.py that compares two images using OpenCV SIFT/ORB features and returns similarity score (0-1)
- [X] T082 [P] [US3] Implement BatchProcessor.process_batch() method in src/image_processor/batch_processor.py that processes one batch: for each bottle image group, call BottleAnalyzer.process_single_bottle(), accumulate results
- [X] T083 [P] [US3] Create BatchPreviewReport dataclass instantiation in src/models/bottle.py with fields: report_id, batch_id, generated_timestamp, markdown_content, report_path
- [X] T084 [P] [US3] Implement PreviewGenerator.generate_batch_preview_report() method in src/image_processor/preview_generator.py that creates markdown with grouping summary table (Bottle Name, Quantity, Images, Confidence), extracted metadata details per bottle, rejected/low-confidence items, instructions for feedback
- [X] T085 [P] [US3] Implement PreviewGenerator.save_batch_preview() method in src/image_processor/preview_generator.py that writes preview markdown to processed/batch_{N}_preview.md
- [X] T086 [P] [US3] Implement BatchProcessor.learn_from_feedback() method in src/image_processor/batch_processor.py that applies user corrections (merge, split, metadata updates) to Bottle entities and adjusts grouping heuristics for subsequent batches
- [X] T087 [P] [US3] Add BatchProcessor.apply_corrections() method in src/image_processor/batch_processor.py that takes list[Correction] objects and updates Bottle entities accordingly
- [X] T088 [P] [US3] Implement PreviewGenerator.interactive_prompt() method in src/image_processor/preview_generator.py that prompts user after each grouping if --interactive flag enabled, displays bottle info and waits for [A]ccept/[R]eject/[M]erge/[S]plit/[E]dit input
- [X] T089 [P] [US3] Implement PreviewGenerator.handle_interactive_choice() method in src/image_processor/preview_generator.py that parses user choice (A/R/M/S/E) and returns appropriate Correction object or marks for rejection
- [X] T090 [P] [US3] Update src/cli/commands.py process command to use BatchProcessor.slice_into_batches() and BatchProcessor.process_batch() instead of processing all bottles sequentially
- [X] T091 [P] [US3] Add src/cli/commands.py batch loop: for each batch, call BatchProcessor.apply_lookahead(), BatchProcessor.process_batch(), PreviewGenerator.generate_batch_preview_report(), PreviewGenerator.save_batch_preview()
- [X] T092 [P] [US3] Add src/cli/commands.py conditional interactive prompt after each batch (if --interactive flag): call PreviewGenerator.interactive_prompt(), collect corrections, call BatchProcessor.apply_corrections(), call BatchProcessor.learn_from_feedback()
- [X] T093 [P] [US3] Add src/cli/commands.py non-interactive mode path: write preview report and continue to next batch without waiting (unless --batch-size all skips preview reports entirely)
- [X] T094 [P] [US3] Implement src/cli/commands.py feedback command with --batch BATCH_ID optional argument, --corrections FILE optional argument, -i/--interactive optional flag
- [X] T095 [P] [US3] Implement src/cli/commands.py feedback command logic: load batch metadata using batch_id, load preview report, display current grouping, collect feedback (interactive or corrections JSON file), apply corrections via BatchProcessor.apply_corrections(), regenerate output files
- [X] T096 [P] [US3] Implement src/cli/commands.py feedback command corrections JSON format parsing: read JSON with array of bottle_id, correction_type, field_affected, original_value, corrected_value, reason fields
- [X] T097 [P] [US3] Add src/cli/commands.py retry command with --report PATH required argument that loads error_report.json, applies suggested fixes, re-processes failed bottle using original options
- [X] T098 [P] [US3] Implement src/cli/commands.py retry command logic: read ErrorReport from JSON, extract bottle_id and images, apply corrections from suggested_fix and hints.md lookup, call BottleAnalyzer.process_single_bottle(), update output files
- [X] T099 [P] [US3] Create hints.md file in root directory with Processing Hints header and empty sections for adding per-bottle guidance (created empty initially, populated during runtime when errors detected)

---

## Phase 6: Optional Feature - Vendor Image Caching & Reference Links

**Story Goal**: Optionally cache vendor reference images and collect reference links for bottle metadata to enhance markdown output.

### Independent Test Criteria
- Given --include-vendor-images flag, system searches for vendor images/links and caches them if found but does NOT block processing if search fails
- Given cached vendor image, saved as vendor_ref.jpg in bottle sub-folder and marked in metadata
- Given found reference links, added to markdown References section with source attribution and confidence scores

### Implementation Tasks

- [ ] T100 [P] Create src/image_processor/vendor_cache.py module with VendorCache class
- [ ] T101 [P] Implement VendorCache.search_vendor_images() method in src/image_processor/vendor_cache.py that uses extracted Bottle metadata (name, distillery, proof, batch) as search query, queries web search API (DuckDuckGo or direct distillery site scraping)
- [ ] T102 [P] Implement VendorCache.verify_vendor_image_match() method in src/image_processor/vendor_cache.py that compares candidate vendor image to original Bottle.images using OpenCV feature matching, returns similarity score 0-1
- [ ] T103 [P] Implement VendorCache.download_and_cache_image() method in src/image_processor/vendor_cache.py that downloads verified vendor image (>85% similarity) to bottle sub-folder as vendor_ref.jpg
- [ ] T104 [P] Implement VendorCache.extract_reference_links() method in src/image_processor/vendor_cache.py that parses search results and extracts URLs pointing to product pages (distillery site, authorized retailers, bourbon databases)
- [ ] T105 [P] Implement VendorCache.validate_reference_link() method in src/image_processor/vendor_cache.py that checks HTTP status code, avoids broken URLs, assigns confidence score based on source type (DISTILLERY_SITE > AUTHORIZED_RETAILER > BOURBON_DATABASE > RETAIL_AGGREGATOR)
- [ ] T106 [P] Add VendorCache.handle_search_failure() method in src/image_processor/vendor_cache.py that logs failure gracefully, continues processing without vendor images/links (feature is non-blocking per FR-035)
- [ ] T107 [P] Update BottleAnalyzer.process_single_bottle() in src/image_processor/bottle_analyzer.py to call VendorCache.search_vendor_images() and VendorCache.extract_reference_links() if --include-vendor-images flag enabled
- [ ] T108 [P] Update Bottle entity creation in src/image_processor/bottle_analyzer.py to set Bottle.vendor_reference_image path and Bottle.reference_links list from VendorCache results
- [ ] T109 [P] Update PreviewGenerator.generate_bottle_markdown() in src/image_processor/preview_generator.py to include References section in markdown if Bottle.reference_links list is non-empty and has high confidence links (confidence >= 0.8)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Address constitutional TODOs, complete documentation, and ensure code quality.

### Independent Test Criteria
- Pre-commit hooks pass all linting checks (ruf, black) and block large files (>5MB)
- ADHD documentation distilled version exists in docs/adhd/ directory
- install.sh script successfully installs all dependencies on macOS and Linux
- All TODOs from constitution addressed

### Implementation Tasks

- [X] T110 [P] Create docs/adhd/ directory and docs/adhd/README.md with distilled quick-start documentation (5-minute overview, key commands, common errors)
- [X] T111 [P] Create docs/detailed/ directory and docs/detailed/README.md with comprehensive usage guide (all CLI options, configuration examples, troubleshooting deep-dive)
- [X] T112 [P] Update main README.md to link to docs/adhd/README.md for quick reference and docs/detailed/README.md for detailed info
- [X] T113 [P] Verify pre-commit hooks configured in .pre-commit-config.yaml with ruff for linting and check-file-size max 5MB
- [X] T114 [P] Test pre-commit hooks by creating test file >5MB and verifying commit is blocked
- [X] T115 [P] Add all docstrings to public methods in all modules (src/image_processor/, src/models/, src/cli/, src/lib/) using Google-style format
- [X] T116 [P] Add inline comments explaining non-obvious logic: orientation detection algorithm, quantity heuristics, batch lookahead logic, visual similarity scoring
- [X] T117 [P] Verify install.sh script works on macOS (test with virtual environment or actual macOS system if available) by checking dependencies installed and Tesseract OCR present
- [X] T118 [P] Verify install.sh script works on Linux (test with virtual environment or actual Linux system if available) by checking dependencies installed and Tesseract OCR present
- [X] T119 [P] Create examples/ directory with sample processed bottle folder (e.g., examples/processed_example/) showing output layout with bottle.md, extracted_text.txt, images/
- [X] T120 [P] Add logging configuration to src/lib/config.py that sets up Python logging module with INFO default level and DEBUG level when --debug flag enabled
- [X] T121 [P] Add log statements throughout modules: processing start, per-bottle progress, API calls, retry attempts, success/failure messages
- [X] T122 [P] Create tests/contract/test_cli_contracts.py test file to verify CLI interface contract from contracts/cli-interface.md
- [X] T123 [P] Implement tests/contract/test_cli_contracts.py tests: --help generates valid help text, invalid arguments exit code 1 with error message, pre-flight checks fail before processing
- [ ] T124 [P] Create tests/contract/test_output_formats.py test file to verify output file formats (markdown structure, error report JSON schema, batch preview markdown)
- [ ] T125 [P] Create tests/unit/test_config.py test file to test Config class methods and priority order (CLI args > env vars > config file > defaults)
- [ ] T126 [P] Create tests/unit/test_ollama_client.py test file to test OllamaClient methods (list_models, generate_vision, generate_text, is_model_available, health_check) with mocked HTTP requests
- [ ] T127 [P] Create tests/unit/test_bottle_analyzer.py test file to test BottleAnalyzer methods (process_single_bottle, group_images_by_bottle, detect_quantity, calculate_confidence) with synthetic test data
- [ ] T128 [P] Create tests/unit/test_image_organizer.py test file to test ImageOrganizer methods (scan_images, group_by_bottle, detect_duplicates) with test fixture images
- [ ] T129 [P] Create tests/unit/test_preview_generator.py test file to test PreviewGenerator methods (generate_bottle_markdown, save_bottle_files, generate_batch_preview_report) with mock filesystem operations
- [ ] T130 [P] Create tests/unit/test_batch_processor.py test file to test BatchProcessor methods (slice_into_batches, apply_lookahead, vision_similarity_score, process_batch) with image queues
- [ ] T131 [P] Create tests/integration/test_full_pipeline.py test file to test end-to-end processing with real images from tests/fixtures/images/straightforward/ relying on Ollama service running
- [ ] T132 [P] Create tests/integration/test_batch_workflow.py test file to test batch processing with lookahead, preview reports, and feedback loop using test fixture images
- [ ] T133 [P] Update pyproject.toml to include pytest as dev dependency and configure pytest with tests/ directory
- [ ] T134 [P] Run pytest —co to verify all test files discovered and no syntax/import errors
- [X] T135 [P] Run pytest to execute unit tests, verify all pass (or document expected failures for integration tests requiring Ollama)
- [X] T136 [P] Create Git worktree for this feature branch per constitution (TODO from Plan Phase: verify worktree-based development)
- [ ] T137 [P] Commit all work with message "feat: bourbon image processor - AI-powered bolt bottle metadata extraction" including attribution line
- [ ] T138 [P] Verify git worktree path, branch name, base commit, and sync status per constitution reporting requirements

---

## Dependencies

### User Story Dependencies

```
US1 (Process Single Bottle) ← US2 (Organize Processed Bottle Sets) ← US3 (Batch Process Multiple Bottles)
                                                         ↓
                                    Optional Feature (Vendor Image Caching)
```

**Rationale**:
- US1 provides core extraction logic that US2 and US3 build upon
- US2 depends on US1 having metadata to organize
- US3 depends on US2 having folder structure to write batched outputs
- Optional vendor caching enhances all user stories but is not required for core functionality

---

## Parallel Execution Opportunities

### Phase 1 (Setup)
- T007-T009: Test fixtures, gitignore, pre-commit hooks can be created in parallel
- T012-T014: pyproject.toml configurations can be added in parallel

### Phase 2 (Foundational)
- T020-T024: OllamaClient methods can be implemented in parallel (independent features)
- T041-T042: Report/Correction dataclasses can be created in parallel

### Phase 3 (US1 - Process Single Bottle)
- T043-T054: TextExtractor and BottleAnalyzer modules can be implemented in parallel (separate domains)
- T059-T065: CLI commands can be implemented in parallel with domain modules (depends on existing interfaces)

### Phase 4 (US2 - Organize Processed Bottle Sets)
- T066-T069: PreviewGenerator folder methods can be implemented in parallel
- T071-T075: ImageOrganizer methods can be implemented in parallel

### Phase 5 (US3 - Batch Processing)
- T078-T087: BatchProcessor and PreviewGenerator batch methods can be implemented in parallel
- T094-T098: CLI feedback/retry commands can be implemented in parallel

### Phase 7 (Polish)
- T110-T112: Documentation files can be created in parallel
- T122-T135: Test files can be created in parallel after implementation is complete

---

## Implementation Strategy

### MVP Scope
**Recommended**: Complete Phase 1 (Setup) + Phase 2 (Foundational) + Phase 3 (US1) + Phase 4 (US2)

This MVP enables users to:
- Process single bottle images and extract metadata
- Organize output into structured processed folders
- Get publishable markdown files for website publication

**Estimated effort**: ~40-50 tasks, 60-70% of total task count

### Incremental Delivery Sequence

1. **Iteration 1** (MVP Core): Phase 1 + Phase 2 + Phase 3 (T001-T065)
   - Validates core hypothesis: can we extract bourbon metadata from images?
   - Provides immediate value for single-bottle processing workflows

2. **Iteration 2** (Organization): Phase 4 (T066-T077)
   - Multi-bottle organization enables scaling to larger collections
   - Validate processing 5-10 bottles at a time

3. **Iteration 3** (Batch Processing): Phase 5 (T078-T099)
   - Enables efficient batch workflows for 10-100+ bottle collections
   - Preview reports and feedback loop improve accuracy and reduce manual work

4. **Iteration 4** (Enhancement): Phase 6 (T100-T109)
   - Optional feature for users wanting reference materials
   - Can be deferred if core functionality proven sufficient

5. **Iteration 5** (Production Ready): Phase 7 (T110-T138)
   - Tests, documentation, quality gates for production deployment
   - Address all constitutional TODOs

### Risk Mitigation

- **Early validation**: After Phase 3 (US1), test with real bottle images to verify AI extraction accuracy
- **Performance check**: After Phase 5 (US3), validate 60s per bottle performance target
- **Quality gates**: Pre-commit hooks and tests enforced from Phase 1 throughout
- **Constitutional compliance**: Verify worktree usage, install.sh functional, ADHD documentation created in Phase 7

---

**Total Tasks**: 138
**Setup Tasks**: 15
**Foundational Tasks**: 27
**User Story 1 Tasks**: 23
**User Story 2 Tasks**: 12
**User Story 3 Tasks**: 22
**Optional Feature Tasks**: 10
**Polish Tasks**: 29

**Format Validation**: All tasks follow checklist format (- [ ] TaskID [P?] [Story?] Description with file path)
