# Sample Images Directory

This directory contains the original images that were processed to generate the bottle metadata in this folder.

## Expected Files

In a real processed output, this directory would contain:

- `IMG_0001.jpg` - Primary bottle image (front view, best angle)
- `IMG_0002.jpg` - Secondary angle shot (back view or side view)
- `detail_label.jpg` - Label close-up detail shot

These images are the source files used for:
- OCR text extraction
- Vision model analysis
- Selection of the primary image for website display
- Verification of extracted metadata

## File Naming Conventions

Original image filenames are preserved from the source folder organization. This allows traceability back to the original photography and supports reprocessing workflows.

## Screenshot Placeholder

In production runs, this directory would contain actual JPEG, PNG, or other image files. The example files are not included here to keep the repository size small.

To see the expected structure, imagine:
```
images/
├── IMG_0001.jpg      (primary bottle shot -清晰的前景照片)
├── IMG_0002.jpg      (alternative angle)
└── detail_label.jpg  (label close-up for better OCR)
```
