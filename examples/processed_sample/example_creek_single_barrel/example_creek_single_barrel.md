# Example Creek Single Barrel

**Status:** NOT publish-ready: not approved by a reviewer

## Needs Review

- Resolved: **batch_number** (batch_number-1): L1,O1 C92 vs user C923 (user value wins)

## Facts

| Field | Value | Confidence | Sources |
|---|---|---|---|
| Name | Example Creek Single Barrel | medium | [1] label "Example Creek Single Barrel" |
| Distillery | Example Creek Distillery | high | [1] label "Example Creek Distillery"; [2] O1 "DISTILLED BY EXAMPLE CREEK DISTILLERY"; [3] W1 "distilled by Example Creek Distillery" |
| Bottler | Unknown: Bottler not part of the label reading; not stated in OCR text or 2 fetched page(s) | - | - |
| Release | Single Barrel | high | [1] label "Single Barrel"; [2] O1 "SINGLE BARREL" |
| Batch | C923 (user-asserted) | high | [5] user |
| Proof | 100 | high | [1] label "100 Proof"; [2] O1 "100 PROOF" |
| ABV | Unknown: Abv not read from label; not stated in OCR text or 2 fetched page(s) | - | - |
| Age | Unknown: Age not read from label; not stated in OCR text or 2 fetched page(s) | - | - |
| Release year | Unknown: Release year not part of the label reading; not stated in OCR text or 2 fetched page(s) | - | - |
| Mash bill | Unknown: Mash bill not part of the label reading; not stated in OCR text or 2 fetched page(s) | - | - |
| Tasting notes | Unknown: Tasting notes not read from label; not stated in OCR text or 2 fetched page(s) | - | - |
| MSRP | Unknown: price on label ($64.99) recorded as retail, not MSRP; 2 page(s) fetched; 1 retail price(s) found (W2 $69.99); no source stated an MSRP | - | - |

## Prices

- **MSRP:** Unknown: price on label ($64.99) recorded as retail, not MSRP; 2 page(s) fetched; 1 retail price(s) found (W2 $69.99); no source stated an MSRP
- **Retail:** $64.99: Price visible on label (retail, not MSRP)
- **Retail:** $69.99: [shop.test](https://shop.test/example-creek-single-barrel), "Our price $69.99", retrieved 2026-11-23

## Description (draft)

Example Creek Single Barrel is distilled by Example Creek Distillery. [1][2]

_Markers number the facts the sentence relies on (not the Sources list): [1] Name, [2] Distillery, [3] Release, [4] Batch, [5] Proof._

## Sources

1. **L1** label reading (vision model) of IMG_0001.jpg
2. **O1** OCR text of IMG_0001.jpg
   - "DISTILLED BY EXAMPLE CREEK DISTILLERY"
   - "SINGLE BARREL"
   - "BATCH C92"
   - "100 PROOF"
3. **W1** [Single Barrel \| Example Creek](https://www.examplecreek.test/single-barrel) (official, examplecreek.test, retrieved 2026-11-23); excerpt: `evidence/W1.txt`
   - "distilled by Example Creek Distillery"
4. **W2** [Example Creek Single Barrel 750ml](https://shop.test/example-creek-single-barrel) (retailer, shop.test, retrieved 2026-11-23); excerpt: `evidence/W2.txt`
   - "Our price $69.99"
5. **U1** user: hint:hints.toml#IMG_0001: Owner read the neck label on the bottle, 2026-11-20

## Verification

- Status: verified
- Model: gpt-oss:120b
- Think: extract high
- Prompt version: facts-1
- Claims accepted: 1; prices accepted: 1; proposals rejected: 1 (listed in the JSON)
- Duration: 74.2 s; model calls: 2

## References

- [Single Barrel \| Example Creek](https://www.examplecreek.test/single-barrel) (official, retrieved 2026-11-23, cited)
- [Example Creek Single Barrel 750ml](https://shop.test/example-creek-single-barrel) (retailer, retrieved 2026-11-23, not cited)

## Web Research Log

- Provider: duckduckgo
- Status: ok
- Searched: 2026-11-23T12:00:00+00:00
- Queries: ""Example Creek Single Barrel" Example Creek Distillery c923 MSRP"; ""Example Creek Single Barrel" Example Creek Distillery Single Barrel"
- web: 4 pages consulted, 1 cited

| # | Site | Class | Retrieved | Outcome | Used for / Reason |
|---|---|---|---|---|---|
| 1 | [examplecreek.test](https://www.examplecreek.test/single-barrel) | official | 2026-11-23 | cited | cited for distillery |
| 2 | [shop.test](https://shop.test/example-creek-single-barrel) | retailer | 2026-11-23 | fetched_not_cited | W2: only a retail price observation |
| 3 | forum.test | - | - | skipped | deny-listed |
| 4 | blog.test | reference | - | skipped | robots.txt disallows fetching |

## Quantity
Total quantity: 1
Total value: Unknown

## Images
1. [IMG_0001.jpg](images/IMG_0001.jpg) (primary)

## Corrections
- **batch_number**: "C92" -> "C923" (hint:hints.toml#IMG_0001: Owner read the neck label on the bottle, 2026-11-20)

## Processing Details
- **Images Processed**: 1
- **Model**: qwen3.6:35b-a3b
- **Extraction Confidence**: HIGH
- **Run**: 20261123_120000
- **Extracted**: 2026-11-23T12:00:00+00:00
- **Warning**: hint overrides label read for batch_number: label 'C92', hint 'C923'
