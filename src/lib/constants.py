"""Project-wide default model names.

A leaf module with no imports, so any layer (CLI, settings, model selector,
analyzers, tests) can read the defaults without import cycles.
"""

#: Default Ollama vision model of the VISION_ANALYSIS stage (reads the label).
DEFAULT_VISION_MODEL = "qwen3.6:35b-a3b"

#: Default Ollama text model of the METADATA_GENERATION stage (fact check:
#: extraction from OCR/web evidence and the verified description).
DEFAULT_TEXT_MODEL = "gpt-oss:120b"
