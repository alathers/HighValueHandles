# Examples Directory

This directory contains sample output from the Bourbon Image Processor to demonstrate expected file layouts and formats.

## Sample: Processed Bottle (`processed_sample/`)

The `processed_sample/` directory shows the typical output structure after processing a bourbon bottle:

```
processed_sample/
└── bottle_001/
    ├── bottle_001.md           # Markdown file for website publishing
    ├── bottle_001.json          # Full metadata with all fields
    ├── bottle_001.txt           # Raw OCR extracted text
    └── images/                  # Original images organized by bottle
        ├── IMG_0001.jpg         # Primary bottle image
        ├── IMG_0002.jpg         # Secondary angle shot
        └── detail_label.jpg     # Label close-up
    └── vendor_image.jpg         # Optional cached vendor photo
```

### File Descriptions

#### `bottle_001.json`
Complete metadata extracted from the bottle images in JSON format. Contains:
- `name`: Bottle name
- `distillery`: Distillery name
- `msrp`: MSRP price (in dollars as a number)
- `proof`: Proof statement (e.g., "107 Proof")
- `age`: Age statement (e.g., "12 Years")
- `batch_number`: Batch number if present
- `specific_release`: Release details (style, series, finish)
- `tasting_notes`: Tasting notes from label
- `full_text`: All extracted OCR text

Example JSON structure:
```json
{
  "name": "Blanton's Single Barrel",
  "distillery": "Buffalo Trace Distillery",
  "msrp": 59.99,
  "proof": "93 Proof",
  "age": "No Age Statement",
  "batch_number": null,
  "specific_release": "Original Single Barrel",
  "tasting_notes": "Caramel, vanilla, and honey on the palate with a long, warm finish.",
  "full_text": "BLANTON'S SINGLE BARREL BOURBON WHISKEY..."
}
```

#### `bottle_001.md`
Formatted Markdown file suitable for website publishing. Contains:
- Product section with name and distillery
- Metadata section (proof, age, batch number, MSRP)
- Release information
- Tasting notes section
- Generated timestamp

#### `bottle_001.txt`
Raw unprocessed OCR text extracted from the label. Useful for:
- Debugging extraction issues
- Manual data entry verification
- Training vision models

#### `images/` subdirectory
All original images that generated this bottle entry. Images are:
- Organized by bottle identity
- Preserved in original format (JPEG, PNG)
- Used as source for batch processing verification

#### `vendor_image.jpg` (optional)
Cached vendor reference image (if available). Used for:
- Verification of extracted data
- Production runs to capture official product images

## Usage

To generate actual output in this format:

```bash
# Process all images in a folder
./process --image-folder /path/to/bottle/photos

# Specify output directory (default: processed/)
./process --image-folder /path/to/bottle/photos --output-dir /custom/output
```

## Integration

The output format is designed for easy integration with:
- Content management systems (WordPress, Ghost, etc.)
- Static site generators (Hugo, Jekyll, etc.)
- Custom bourbon inventory databases
- YouTube channel discussion preparation

All files use the same bottle identifier prefix for easy association and batch processing.
