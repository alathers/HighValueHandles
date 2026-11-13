"""Data model types for bourbon inventory management."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bottle import Bottle, Correction, ExtractedMetadata, Quantity, ReferenceLink
from .image import Image
from .processing import BatchPreviewReport

__all__ = [
    "Image",
    "ExtractedMetadata",
    "Bottle",
    "ReferenceLink",
    "Quantity",
    "BatchPreviewReport",
    "Correction",
]
