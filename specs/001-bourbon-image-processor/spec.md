# Feature Specification: Bourbon Image Processor

**Feature Branch**: `001-bourbon-image-processor`

**Created**: 2026-11-09

**Status**: Draft

**Input**: User description: "Identify the Bottle, the MSRP, the Distillery, The specific release information, The tasting notes for this specifc bottle. Each bottle should get a Markdown file with it's own information

Once processing is done, make a folder called processed. In that folder make a sub-folder for each set of images that belong together (multiple angles for the same bottle, box or other item).  Include all of the output files for that as well.

The target audience is creating data files that I will use to publish to my website for a youtube channel discussion burboun, so ensure the text covers all the details the audiance may be interested in."

## Clarifications

### Session 2026-11-09

- Q: What should happen when a single bottle fails to process completely? → A: Retry up to 3 times, then produce an error report for retry mechanism once repaired, plus a hints.md file for hard-to-handle items
- Q: How does the system determine which images belong to the same bottle vs. different bottles? → A: Complex grouping: exact duplicates track as quantity; similar sequential photos assume distinct items; batch number differences create unique bottles; quantity always tracked (even if 1); user provides no duplicate angles of same bottle
- Q: How should the test/validation feedback loop work? → A: Per-batch incremental: Process in small batches, generate preview reports between batches, user provides feedback, system learns and adjusts before next batch
- Q: What format should the preview report use for user feedback? → A: Hybrid: Primary markdown with tables for editorial comfort and version control, with optional interactive CLI prompts for faster review workflow
- Q: What is the default batch size for processing and preview report generation? → A: Default 5 bottles per batch with lookahead of next 3 images to catch multi-angle photos that belong to same batch; configurable via CLI argument including "all" for full-folder processing
- Q: Should reference materials found during bottle research include links to vendor pages? → A: Yes, include links to any reference materials found, especially vendor pages about their own product with focus on pages explicitly about the specific bottle/release

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Process Single Bottle Images (Priority: P1)

A user has one or more images of a bourbon bottle/box (front, back, side, label close-ups, etc.) and wants to extract comprehensive information about the bottle including: name, MSRP, distillery, release details (proof, age, batch), and tasting notes. The system should process these images and generate a well-formatted markdown file containing all relevant information ready for website publication and YouTube video discussion preparation.

**Why this priority**: This is the core value proposition - extracting bourbon bottle information from images into publishable data. Without this, no other workflows are possible.

**Independent Test**: Can be fully tested by providing a set of bottle images and verifying the system generates a markdown file with complete, accurate bottle information without requiring batch processing or folder organization.

**Acceptance Scenarios**:

1. **Given** one or more images of a bourbon bottle/box, **When** the user runs the processor, **Then** the system analyzes the images and returns a markdown file containing: bottle name, MSRP, distillery, release information, and tasting notes
2. **Given** bottle images with legible label information, **When** the user processes them, **Then** the extracted information includes proof, age, batch/lot number if present on labels
3. **Given** bottle images with pricing information, **When** the user processes them, **Then** the system identifies and extracts the MSRP from tags, labels, or receipts
4. **Given** bottle images that are rotated or incorrectly oriented, **When** the user processes them, **Then** the system auto-orient images and extracts text correctly from labels
5. **Given** a completed extraction, **When** the user checks the bottle folder, **Then** they find a human-readable text file containing all raw extracted text from all images
3. **Given** bottle images with pricing information, **When** the user processes them, **Then** the system identifies and extracts the MSRP from tags, labels, or receipts
4. **Given** images of a limited release or special edition bottle, **When** the user processes them, **Then** the release information includes special designation, release number, series name, or other relevant details from packaging
5. **Given** a bottle with tasting notes on packaging or from external sources, **When** the user processes them, **Then** the markdown file includes tasting notes describing flavor profile, nose, palate, finish

---

### User Story 2 - Organize Processed Bottle Sets (Priority: P2)

A user has processed multiple bottles and needs the images and output files organized in a structured way. The system should create a "processed" folder with sub-folders for each bottle, containing all related images and the generated markdown file for that specific bottle.

**Why this priority**: Without organization, users cannot manage a growing collection of processed bottles. This enables workflow scalability.

**Independent Test**: Can be tested by processing multiple distinct bottles and verifying each bottle has its own sub-folder containing all its images and markdown file, without requiring batch automation.

**Acceptance Scenarios**:

1. **Given** images two or more different bottles have been processed, **When** the system completes processing, **Then** each bottle's images and markdown file are placed in separate sub-folders within the "processed" directory
2. **Given** multiple images of the same bottle (front, back, label close-ups), **When** the system processes them, **Then** all images for that bottle are saved together in the same sub-folder
3. **Given** a bottle ID or name during processing, **When** the system creates folders, **Then** the sub-folder name clearly identifies the bottle (e.g., using bottle name or batch number)
4. **Given** a "processed" folder already exists, **When** new bottles are processed, **Then** the system adds new sub-folders without overwriting existing ones

---

### User Story 3 - Batch Process Multiple Bottles (Priority: P3)

A user has many bottles and wants to efficiently process a collection of images at once, maintaining organized output for each bottle. The system should accept a collection of image sets and process each set independently, generating organized output for all bottles. Processing occurs in batches with preview reports for feedback between batches.

**Why this priority**: Optimizes workflow for users managing larger inventories, but not required for initial value.

**Independent Test**: Can be tested by providing multiple pre-organized bottle image sets and verifying each set is processed correctly with output in separate folders.

**Acceptance Scenarios**:

1. **Given** multiple folders each containing images for a different bottle, **When** the user runs batch processing, **Then** the system processes each folder independently and creates corresponding sub-folders in "processed"
2. **Given** a selection of individual images representing multiple bottles, **When** the user processes them, **Then** the system groups related images by bottle and processes each group separately
3. **Given** batch processing runs, **When** any single bottle processing fails, **Then** the system continues processing remaining bottles and reports errors for failed items
4. **Given** batch processing completes or reaches batch size threshold, **When** the system generates a preview report, **Then** it creates a markdown file with tables showing grouping decisions and extracted metadata for user review
5. **Given** optional interactive CLI mode enabled, **When** preview report is generated, **Then** the system displays each grouping and allows user to accept/reject/modify inline via CLI prompts
6. **Given** user reviews preview report (markdown or CLI) and flags incorrect groupings or extractions, **When** user provides feedback, **Then** the system re-processes only flagged items and learns from corrections for remaining batches
7. **Given** a bottle is processed and vendor reference images are available, **When** a matching vendor image is found (same name, distillery, proof, batch), **Then** it is optionally cached in the bottle's processed sub-folder and marked as a vendor reference
8. **Given** batch processing reaches default size of 5 bottles, **When** preview report is generated, **Then** the system looks ahead at up to 3 subsequent images to check for multi-angle photos of bottles in current batch before stopping
9. **Given** user specifies custom batch size via CLI argument, **When** processing runs, **Then** the system uses the custom size for preview report generation instead of default 5
10. **Given** user specifies "all" as CLI batch argument, **When** processing runs, **Then** the system processes all images in folder without stopping for preview reports

---

### Edge Cases

- What happens when bottle label text is illegible, damaged, or absent from images? System should attempt partial extraction and mark missing information clearly
- How does system handle bottles with multiple MSRPs (e.g., different retailers/regions)? Extract or note all visible prices without choosing
- What happens when images are rotated 90°, 180°, or 270° from upright? System should auto-orient images before text extraction
- How does system handle images with inverted or sideways text (text printed at angles on labels)? Extract and orient text for human readability regardless of source orientation
- What happens when distillery information is inconsistent across label sources (choice, ETL vs. DSP)? Capture all listed information with source attribution
- How does system handle bottles with no clear tasting notes in images? Generate placeholder or note that tasting notes need manual addition
- What happens when images don't contain a complete bottle (only label, only cap, etc.)? Process available information and note limitations
- What happens when folder naming conflicts occur (two bottles with same name)? Use disambiguating info like batch/proof/folder timestamp
- How does system handle non-bourbon whiskey images included by mistake? Process as whiskey and flag type detection, or skip with clear error message
- What happens when bottle processing fails completely after retry attempts? Generate error report file for retry mechanism and add entry to hints.md for manual intervention guidance
- How does system distinguish between identical duplicate bottles and different bottles that look similar? Use batch numbers, release details, and visual analysis - sequential similar photos assumed distinct unless identified as identical through batch/proof matching
- What happens when images show multiple identical bottles in a single frame (e.g., 2 boxes side-by-side)? Track as quantity=2 with a single bottle entry, include all images for quantity documentation
- How does system handle uncertainty about whether two similar photos are duplicates or different bottles? Default to treating as distinct items; user can manually merge if needed

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST identify and extract the bourbon bottle name from label, packaging, or image context
- **FR-002**: System MUST extract MSRP information from price tags, shelf tags, receipts, or labels where present
- **FR-003**: System MUST identify the distillery that produced the bottled bourbon (including both legal owner/DSP and non-distilling producer status if applicable)
- **FR-004**: System MUST extract specific release information including: proof/ABV, age statement, batch/lot number, barrel number, and special release designations
- **FR-005**: System MUST extract tasting notes from packaging, labels, or identify where tasting notes may be obtained
- **FR-006**: System MUST generate a markdown file for each processed bottle containing all extracted information
- **FR-007**: Markdown file structure MUST include sections for: Bottle Name, MSRP, Distillery, Release Details, and Tasting Notes
- **FR-008**: System MUST organize processed output into a "processed" folder
- **FR-009**: System MUST create a sub-folder for each unique bottle within "processed"
- **FR-010**: System MUST save all images for a specific bottle together with that bottle's markdown file in its sub-folder
- **FR-011**: System MUST group multiple images belonging to the same bottle (different angles, packaging elements) together
- **FR-012**: System MUST name sub-folders with clearly identifiable bottle information
- **FR-033**: System MAY attempt to find and cache a vendor reference image (official product photo) for each bottle if one can be located and verified as matching the exact bottle (same name, distillery, proof, batch)
- **FR-034**: If a vendor image is found and cached, it MUST be saved in the bottle's processed sub-folder and marked as a vendor reference image in the metadata
- **FR-035**: Vendor image caching is optional - caching failure or inability to locate a vendor image MUST NOT prevent bottle processing from completing
- **FR-040**: When researching bottle information, system MUST find and include links to reference materials, especially vendor pages about their own product, focusing on pages explicitly about the specific bottle/release
- **FR-041**: If reference link(s) are found, they MUST be included in the generated markdown file in a dedicated "References" section
- **FR-013**: Output markdown content MUST be suitable for website publication (well-formatted, audience-appropriate detail level)
- **FR-014**: System MUST handle cases where some information cannot be extracted by marking it as missing or [NEEDS MANUAL REVIEW]
- **FR-015**: System MUST save all raw extracted text in a human-readable file within each bottle's sub-folder
- **FR-016**: System MUST auto-detect and correct image orientation (rotation) before text extraction
- **FR-017**: Extracted text file MUST preserve text as read from images without transformation (e.g., vertical text kept vertical, rotated text preserved as-is)
- **FR-018**: System MUST automatically retry failed bottle processing up to 3 times before treating as failed
- **FR-019**: System MUST generate an error report file for each failed bottle attempt, suitable for use with a retry mechanism once issues are repaired
- **FR-020**: System MUST create and maintain a hints.md file documenting guidance for hard-to-handle bottles, including specific challenges and suggested manual intervention approaches
- **FR-021**: System MUST detect and track quantity of identical items (e.g., 2 bottles of same name/proof/batch) and create a single set of artifacts with quantity metadata
- **FR-022**: System MUST assume sequential similar photos are different items (unless proven identical) and handle potential duplicates conservatively
- **FR-023**: System MUST identify batch numbers and other distinguishing features from labels to differentiate similar-looking bottles
- **FR-024**: System MUST always include quantity field in bottle metadata, with default value of 1 for single items
- **FR-025**: System MUST group all images of identical items (including quantity-scenarios of multiple bottles in one image or duplicate image sets) together in the same sub-folder
- **FR-026**: System MUST process bottles in batches and generate preview reports after each batch containing grouping decisions and extracted metadata for user review
- **FR-027**: System MUST support user feedback on preview reports to flag incorrect groupings or extraction errors
- **FR-028**: System MUST re-process only flagged items from user feedback without requiring full re-processing of all bottles
- **FR-029**: System MUST learn from user feedback and apply improvements to subsequent batches (e.g., adjust grouping heuristics, extraction patterns)
- **FR-030**: Preview report format MUST clearly display each bottle's grouping decisions and key metadata for quick validation by user
- **FR-031**: Preview report MUST be generated as a markdown file with formatted tables showing bottle groups, images, and extracted metadata
- **FR-032**: System MUST support optional interactive CLI mode for preview review, allowing user to accept/reject/modify grouping and extraction decisions inline
- **FR-036**: System MUST use default batch size of 5 bottles for processing and preview report generation
- **FR-037**: System MUST perform lookahead check of up to 3 subsequent images to determine if they are multi-angle photos of bottles in current batch (same bottle identity), including them in current batch if they match
- **FR-038**: Batch size MUST be configurable via CLI argument allowing user to override default value
- **FR-039**: CLI argument MUST support "all" keyword to process all files in folder without batch stopping for preview reports

### Key Entities

- **Image**: Visual input file representing one view of a bourbon bottle, packaging element, or label. May be front, back, side, close-up of label text, box, receipts, etc. User does not capture duplicate angles of the same bottle.
- **Bottle**: Complete bourbon product entity consisting of one or more related images, containing all metadata (name, distillery, MSRP, release info, tasting notes, quantity)
- **Bottle Set**: Group of images that represent the same physical bottle identity (same name, distillery, proof, batch). May include multiple images with identical content (indicating quantity > 1) or images containing multiple identical bottles in one frame.
- **Extracted Metadata**: Structured information derived from images including: bottle name, distillery, MSRP, release information, tasting notes, quantity
- **Vendor Image**: Optional official product photo for the exact bottle matching the extracted metadata (name, distillery, proof, batch). Cached if available and verifiable.
- **Markdown File**: Human-readable output document containing extracted metadata, organized by sections, suitable for website publication
- **Quantity**: Count of identical items detected across all images for a bottle set. Always tracked, default is 1. Derived from: identical duplicate images, images showing multiple identical bottles in one frame, or explicit count tracking

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can extract complete bourbon bottle information from images and obtain a publishable markdown file within 60 seconds per bottle set
- **SC-002**: Extracted information accuracy (bottle name, distillery, proof, age) achieves 95%+ correctness on clearly labeled bottles
- **SC-003**: System successfully organizes 100% of processed bottle sets into separate sub-folders with proper file grouping
- **SC-004**: Generated markdown files require manual editing or review for less than 10% of bottles with clear, legible labels
- **SC-005**: Users can process and organize a 10-bottle collection in under 10 minutes
- **SC-006**: Each extracted metadata field (name, MSRP, distillery, release info, tasting notes) is present in 90%+ of output files for bottles where information is available on images

## Assumptions

- Images are of sufficient resolution and quality to read printed text on labels and packaging
- User provides images that contain relevant labeling information (bottles without labels may have limited extraction)
- Target audience (YouTube bourbon discussion viewers) is interested in: brand/name, proof/age/distillery backstory, MSRP/value discussion, tasting profile, and rarity/collectibility factors
- Processing may involve optical character recognition (OCR) and visual analysis of bottle/label features
- "Processed" folder will be created in the working directory or location specified by user
- Output markdown files should be comprehensive but concise enough for website publication (not raw field dumps)
- System operates in a local environment without requiring cloud services unless user explicitly configures otherwise
- Tasting notes may come from packaging text or may need placeholder directions for where user can source them (distillery website, tasting guides)
