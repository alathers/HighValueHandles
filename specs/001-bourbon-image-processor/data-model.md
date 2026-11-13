# Data Model: Bourbon Image Processor

**Feature**: Bourbon Image Processor
**Date**: 2026-11-09
**Status**: Complete

## Overview

This document defines the entities, relationships, and validation rules for the bourbon image processing system. The data model supports image grouping, quantity detection, metadata extraction, and reference link management.

---

## Core Entities

### Image

Represents a single image file captured by the user. Images may show one or more bottles, labels, boxes, receipts, or other artifacts.

**Attributes**:
- `filepath` (string, required): Absolute path to original image file
- `filename` (string, required): Original filename (e.g., "IMG_0001.jpeg")
- `file_size_bytes` (integer, required): Size of file in bytes
- `format` (string, required): Image format (JPEG, PNG, etc.)
- `width_px` (integer, required): Image width in pixels
- `height_px` (integer, required): Image height in pixels
- `original_orientation` (enum, required): Detected orientation before correction (UPRIGHT, ROTATED_90, ROTATED_180, ROTATED_270, UNKNOWN)
- `corrected` (boolean, required): Whether orientation correction was applied
- `ocr_text` (string, optional): Raw extracted text from image
- `detected_text_regions` (array, optional): List of bounding boxes where text was detected
- `is_duplicate` (boolean, optional): Whether this image is an exact duplicate of another
- `duplicate_of_id` (string, optional): Original image ID if this is a duplicate

**Validation Rules**:
- `filepath` must exist and be readable
- `filepath` extension must be supported image format (.jpeg, .jpg, .png, .tiff, .heic)
- `width_px` and `height_px` must be positive integers
- `original_orientation` must be one of defined enum values
- `format` must match file extension

**Constraints**:
- Images are immutable once captured (filepath never changes)
- Original file is preserved in output folder (no modifications to source)

---

### Bottle

Complete bourbon product entity representing one or more related images with metadata extracted from them.

**Attributes**:
- `id` (string, required): Unique identifier (e.g., hash of name + distillery + proof + batch)
- `name` (string, required): Extracted bottle name (e.g., "Buffalo Trace Bourbon")
- `distillery` (string, required): Distillery name including ownership status (e.g., "Buffalo Trace Distillery (DSP-KY-1)")
- `msrp` (decimal, optional): Manufacturer suggested retail price
- `msrp_currency` (string, optional): Currency code (default: USD)
- `proof` (decimal, optional): Alcohol proof (e.g., 90.0)
- `abv` (decimal, optional): Alcohol by volume (derived from proof: proof / 200)
- `age_years` (integer, optional): Age statement in years (e.g., 8)
- `age_description` (string, optional): Verbal age description (e.g., "No Age Statement", "Aged 6-8 years")
- `batch_number` (string, optional): Batch/lot identifier (e.g., "BT24-07", "A123")
- `barrel_number` (string, optional): Barrel identifier (e.g., "12-3456")
- `proof_variation` (string, optional): Notes about proof variance (e.g., "varies by barrel")
- `special_designation` (string, optional): Special release designation (e.g., "Single Barrel", "Barrel Strength")
- `year_released` (integer, optional): Release year (e.g., 2024)
- `release_number` (string, optional): Sequential release identifier (e.g., "Limited Release #7")
- `series_name` (string, optional): Series name (e.g., "Kentucky Owen County", "Eagle Rare 17 Year")
- `tasting_notes` (string, optional): Tasting notes describing nose, palate, finish
- `quantity` (integer, required): Number of identical items detected (default: 1)
- `images` (array of Image, required): All images related to this bottle
- `vendor_reference_image` (string, optional): Path to cached vendor reference image
- `reference_links` (array of ReferenceLink, optional): Links to reference materials
- `creation_timestamp` (datetime, required): When this bottle record was created
- `processing_status` (enum, required): COMPLETE, PARTIAL, FAILED

**Validation Rules**:
- `name`, `distillery` are required
- `quantity` must be positive integer (≥ 1)
- If `proof` is provided, `abv` must be calculated as `proof / 200`
- `batch_number` required for distinguishing bottles with same name/distillery/proof
- `processing_status` must be COMPLETE or PARTIAL to include in output

**Constraints**:
- Two bottles with identical `name`, `distillery`, `proof`, and `batch_number` are considered the same bottle
- `quantity` increments when duplicate images or multiple bottles detected, bottle metadata merges
- If `processing_status` is PARTIAL, fields marked as `[EXTRACTION FAILED]` or `[MISSING]` must be documented in error report

---

### Bottle Set

Group of one or more bottles that were processed together in a batch. Used for preview reports and feedback tracking.

**Attributes**:
- `batch_id` (string, required): Unique batch identifier
- `batch_number` (integer, required): Numerical batch index (1-indexed)
- `bottles` (array of Bottle, required): Bottles in this batch
- `preview_generated` (boolean, required): Whether preview report was generated
- `user_feedback_received` (boolean, required): Whether user provided feedback on this batch
- `feedback_timestamp` (datetime, optional): When feedback was received
- `corrections_applied` (array of Correction, optional): Corrections made based on user feedback

**Validation Rules**:
- `batch_number` must be positive integer
- `bottles` array cannot be empty
- `corrections_applied` valid only if `user_feedback_received` is true

---

### Extracted Metadata

Structured information derived from bottle images. This is the output entity that populates the Bottle entity.

**Attributes**:
- `source_image_id` (string, required): ID of image this metadata was extracted from
- `bottle_name` (string, optional): Extracted bottle name
- `confidence_bottle_name` (float, optional): Confidence score (0-1) for name extraction
- `distillery` (string, optional): Extracted distillery name
- `confidence_distillery` (float, optional): Confidence score for distillery extraction
- `msrp` (decimal, optional): Extracted MSRP
- `confidence_msrp` (float, optional): Confidence score for MSRP extraction
- `proof` (decimal, optional): Extracted proof/ABV
- `confidence_proof` (float, optional): Confidence score for proof extraction
- `age` (string, optional): Extracted age statement
- `confidence_age` (float, optional): Confidence score for age extraction
- `batch_number` (string, optional): Extracted batch/lot number
- `confidence_batch` (float, optional): Confidence score for batch extraction
- `special_designation` (string, optional): Extracted special designation
- `tasting_notes` (string, optional): Extracted tasting notes
- `confidence_tasting` (float, optional): Confidence score for tasting notes

**Validation Rules**:
- Confidence scores, if provided, must be between 0 and 1 inclusive
- At least one field with confidence > 0.5 is required for metadata to be useful

---

### Vendor Image

Optional official product image for a specific bottle, cached from vendor websites.

**Attributes**:
- `source_url` (string, required): URL where image was found
- `local_path` (string, required): Path to cached copy in bottle folder
- `source_type` (enum, required): DISTILLERY_SITE, RETAILER_LISTING, BOURBON_DATABASE
- `similarity_score` (float, optional): Match confidence vs. actual bottle images (0-1)
- `verified` (boolean, required): Whether image was verified as matching the exact bottle
- `cached_timestamp` (datetime, required): When image was downloaded and cached

**Validation Rules**:
- `source_url` must be valid HTTP/HTTPS URL
- `similar_score`, if provided, must be between 0 and 1 inclusive
- `similarity_score` >= 0.85 required for `verified=true`

**Constraints**:
- Vendor image caching is optional; missing vendor image does not block processing
- If `verified=false`, `vendor_reference_image` in Bottle entity should be None

---

### ReferenceLink

Link to external reference material (vendor pages, distillery product pages, retailer listings).

**Attributes**:
- `url` (string, required): Link URL
- `title` (string, required): Link title (e.g., "Buffalo Trace Bourbon - Distillery Product Page")
- `source` (enum, required): DISTILLERY_SITE, AUTHORIZED_RETAILER, BOURBON_DATABASE, RETAIL_AGGREGATOR
- `confidence` (float, required): Confidence that link is specifically about this bottle/release (0-1)
- `bottle_match_evidence` (string, optional): Evidence that matches this link to the bottle (e.g., "Contains matching batch number BT24-07")
- `priority` (integer, required): Display priority (lower = higher priority, default: 10)

**Validation Rules**:
- `url` must be valid HTTP/HTTPS URL
- `confidence` must be between 0 and 1 inclusive
- `priority` must be positive integer (default: 10)

**Constraints**:
- Links with `confidence` >= 0.8 are included in markdown References section
- Links with `confidence` < 0.8 may be included with "[Unverified]" qualifier
- If no reference links with `confidence` >= 0.7, References section may be omitted

---

### Quantity

Model for tracking quantity of identical bottles across images.

**Attributes**:
- `bottle_id` (string, required): Reference to Bottle entity
- `quantity` (integer, required): Count of identical items (default: 1)
- `detection_method` (enum, required): EXACT_DUPLICATE_IMAGES, MULTIPLE_IN_ONE_FRAME, USER_SPECIFIED, INFERRED
- `supporting_image_id` (string, optional): Image providing evidence for quantity (e.g., image showing 2 boxes)
- `confidence` (float, optional): Confidence in quantity detection (0-1)

**Validation Rules**:
- `quantity` must be positive integer (≥ 1)
- If `detection_method` is EXACT_DUPLICATE_IMAGES, `supporting_image_id` may be None (duplicate check is hash-based)

**Constraints**:
- Quantity detection is conservative: favors multiple bottles over inferred duplicates
- `quantity` in Bottle entity is set from this entity's `quantity` field

---

### BatchPreviewReport

Markdown document generated after each batch for user review.

**Attributes**:
- `report_id` (string, required): Unique identifier
- `batch_id` (string, required): Reference to Bottle Set
- `generated_timestamp` (datetime, required): When report was generated
- `markdown_content` (string, required): Full markdown text of report
- `report_path` (string, required): Path to saved report file

**Structure** (markdown):
```markdown
# Batch {N} Preview Report

Generated: {timestamp}

## Grouping Summary
| Bottle Name | Quantity | Images | Confidence |
|-------------|----------|--------|------------|
| Buffalo Trace | 2 | IMG_001.jpg, IMG_002.jpg | High |
| Blanton's | 1 | IMG_003.jpg | Medium |

## Extracted Metadata

### Buffalo Trace (Quantity: 2)
**Distillery**: Buffalo Trace Distillery (DSP-KY-1)
**Proof**: 90 proof
**MSRP**: $29.99
**Batch**: BT24-07

{...details...}

## Rejected / Low-Confidence Items
- IMG_004.jpg → Unable to identify bottle; manual review required

## Instructions
To provide feedback, run: python -m cli.commands feedback --batch {batch_id}
```

---

### Correction

User-provided correction applied to a bottle after preview review.

**Attributes**:
- `correction_id` (string, required): Unique identifier
- `bottle_id` (string, required): Bottle being corrected
- `correction_type` (enum, required): MERGE_BOTTLES, SPLIT_BOTTLES, UPDATE_METADATA, CORRECT_GROUPING, ADD_IMAGE, REMOVE_IMAGE
- `field_affected` (string, optional): Which field was updated (e.g., "batch_number", "distillery")
- `original_value` (string, optional): Value before correction
- `corrected_value` (string, optional): Value after correction
- `reason` (string, optional): User note explaining correction
- `applied_timestamp` (datetime, required): When correction was applied

**Validation Rules**:
- `correction_type` must be one of defined enum values
- If `correction_type` is UPDATE_METADATA, `field_affected`, `original_value`, and `corrected_value` are required

---

### ErrorReport

Report generated when bottle processing fails after all retries.

**Attributes**:
- `error_id` (string, required): Unique identifier
- `bottle_id` (string, required): Bottle that failed
- `images` (array of Image, required): Images being processed when error occurred
- `error_type` (enum, required): OCR_FAILURE, MODEL_TIMEOUT, DECODING_ERROR, NETWORK_ERROR, ORIENTATION_DETECTION_FAILED
- `error_message` (string, required): Human-readable error description
- `retry_attempts` (integer, required): Number of retry attempts before failure
- `elapsed_time_seconds` (decimal, required): Total time spent processing before failure
- `suggested_fix` (string, optional): Suggested remediation
- `hint_topic` (string, optional): Keywords for hints.md lookup

**Structure** (JSON):
```json
{
  "error_id": "err_abc123",
  "bottle_id": "bottle_def456",
  "images": ["IMG_001.jpeg"],
  "error_type": "OCR_FAILURE",
  "error_message": "Tesseract unable to extract text from rotated image after 3 attempts",
  "retry_attempts": 3,
  "elapsed_time_seconds": 58.4,
  "suggested_fix": "Manually rotate image using photo editor, or capture new straight-on photo",
  "hint_topic": "rotated_images"
}
```

---

## Relationships

```
BottleSet (1) ────────(*) Bottle
Bottle (1) ──────────(*) Image
Bottle (1) ──────────(1) Quantity
Bottle (1) ──────────(0..1) VendorImage
Bottle (1) ──────────(0..*) ReferenceLink
Image (1) ──────────(0..1) ExtractedMetadata
BottleSet (1) ────────(1) BatchPreviewReport
BottleSet (1) ────────(0..*) Correction
```

---

## State Transitions

### Bottle Processing Status

```
PENDING → PROCESSING → COMPLETE
PENDING → PROCESSING → PARTIAL
PENDING → PROCESSING → FAILED

PARTIAL → RETRYING → COMPLETE
PARTIAL → RETRYING → PARTIAL
PARTIAL → RETRYING → FAILED
```

### Batch Processing Flow

```
BATCH_PENDING → PROCESSING → PREVIEW_GENERATED
PREVIEW_GENERATED → AWAITING_FEEDBACK → COMPLETED
PREVIEW_GENERATED → AWAITING_FEEDBACK → CORRECTIONS_APPLIED → COMPLETED
PREVIEW_GENERATED → (no user feedback) → COMPLETED (default after timeout)
```

---

## Validation Summary

**Entity-Level Validation**:
- Fields marked as `required` must be non-null/empty
- Enum fields must use predefined values
- Confidence scores, if present, must be 0-1 inclusive
- Positive constraints (e.g., `quantity` ≥ 1) enforced

**Cross-Entity Validation**:
- Bottle uniqueness: Same `name`, `distillery`, `proof`, `batch_number` = same bottle
- Image grouping: Images in same `Bottle` must have related visual/text similarity
- Reference link verification: `VendorImage.verified` requires `similarity_score` ≥ 0.85

**Data Integrity Rules**:
- Original images never modified
- Output files saved to separate `processed/` directory
- Error reports contain all information needed to retry
- Preview reports are complete before batch finalization

---

## Output File Formats

### Bottle Metadata (Markdown)

File: `processed/{bottle_name}_{timestamp}/bottle.md`

```markdown
# {Bottle Name}

## Distillery
{Distillery Name}

## MSRP
${MSRP}

## Release Information
- **Proof**: {X} proof
- **ABV**: {X}%
- **Age**: {X} years
- **Batch**: {Batch Number}
- **Barrel**: {Barrel Number} (if applicable)
- **Special Designation**: {e.g., Single Barrel}

## Tasting Notes

{Nose, palate, finish description}

## References
- [Distillery Product Page](URL) - Official distillery information
- [Retailer Listing](URL) - Product details and availability

## Quantity
{N} bottles detected

## Images
{List of original images}

## Processing Notes
{Any warnings, constraints, or notes about extraction quality}
```

### Extracted Text (Plain Text)

File: `processed/{bottle_name}_{timestamp}/extracted_text.txt`

```
Raw extracted text from all images, preserving orientation and format
as captured by OCR and vision model processing.
```

---

## Summary

The data model supports:
- Image grouping and quantity detection
- Metadata extraction with confidence scoring
- Optional vendor image caching with verification
- Reference link collection for external sources
- Batch processing with preview reports and user feedback
- Error handling with retry mechanisms and hints.md

All entities are designed to enable the functional requirements defined in the feature specification while supporting the performance goals (60s per bottle, 95% accuracy).
