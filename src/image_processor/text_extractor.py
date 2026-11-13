"""Text extraction from bourbon bottle images using Tesseract OCR.

Detects orientation, corrects it, extracts text from labeled regions,
and handles errors with automatic retry logic.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from lib.error_handling import RetryManager
from lib.exceptions import ImageProcessingError, OCRError, RetryExhaustedError

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from bourbon bottle images using Tesseract OCR.

    Handles orientation detection, text region selection, and retry logic.

    Attributes:
        retry_manager: RetryManager for automatic retries on failures.
    """

    def __init__(self, retry_manager: Optional[RetryManager] = None):
        """Initialize text extractor.

        Args:
            retry_manager: Optional RetryManager for automatic retries.
        """
        self.retry_manager = retry_manager or RetryManager()

    def extract_text(
        self,
        image_path: Path,
        orientation_corrected: bool = False,
        select_regions: bool = True,
    ) -> str:
        """Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to image file.
            orientation_corrected: Whether image is already upright.
            select_regions: Whether to select text regions before OCR.

        Returns:
            Extracted text as string.

        Raises:
            FileNotFoundError: If image file not found.
            ImageProcessingError: If image cannot be loaded or processed.
            OCRError: If OCR text extraction fails after retries.
        """
        # Load image
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image {image_path}", str(e))

        # Correct orientation if needed
        if not orientation_corrected:
            orientation = self.detect_orientation(np.array(image))
            if orientation != 0:
                # Rotate image to upright
                image = image.rotate(-orientation, expand=True)
                logger.info(f"Rotated image {image_path} by {-orientation} degrees")

        # Select text regions if enabled
        if select_regions:
            text_regions = self.select_text_regions(np.array(image))
            if text_regions:
                # Extract text from each region and concatenate
                extracted_parts = []
                for region in text_regions:
                    region_image = Image.fromarray(region)
                    text = self._perform_ocr(region_image)
                    if text.strip():
                        extracted_parts.append(text.strip())
                return " ".join(extracted_parts)

        # Extract text from full image
        return self._perform_ocr(image)

    def detect_orientation(self, image: np.ndarray) -> int:
        """Detect image orientation using OpenCV.

        Args:
            image: Image as numpy array.

        Returns:
            Rotation needed to make image upright (0, 90, 180, or 270 degrees).

        Raises:
            ImageProcessingError: If orientation detection fails.
        """
        try:
            # Convert to grayscale for improved text detection accuracy
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Primary method: Use Tesseract OSD (Orientation and Script Detection)
            # OSD analyzes text layout to determine rotation angle
            # PSM 0 = Orientation and script detection (OSD) only
            # min_characters_to_try=10 ensures enough text for reliable detection
            try:
                osd = pytesseract.image_to_osd(
                    gray,
                    config="--psm 0 -c min_characters_to_try=10",
                )
                # Parse OSD result to get orientation
                for line in osd.split("\n"):
                    if line.startswith("Orientation in degrees:"):
                        return int(line.split(":")[1].strip())
            except Exception as e:
                logger.warning(f"OSD detection failed: {e}")

            # Fallback: Detect text skew using minimum area rectangle
            # Used when OSD fails (e.g., insufficient text on image)
            coords = np.column_stack(np.where(gray > 0))
            if len(coords) > 0:
                # Get the angle of the rotated bounding box that fits all pixels
                # cv2.minAreaRect returns (center, (width, height), angle)
                # [-1] extracts just the angle value
                angle = cv2.minAreaRect(coords)[-1]

                # Normalize angle: OpenCV returns angle in range [-90, 0]
                # Convert to [-90, 90] range for easier normalization
                # If angle < -45, it's actually the complement, so adjust:
                # e.g., -80° becomes -(90 + (-80)) = -10°
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle

                # Round to nearest 90-degree increment (standard image rotations)
                # This handles minor skew errors and gives clean 0/90/180/270 results
                angle = round(angle / 90) * 90
                return int(angle) % 360

            return 0
        except Exception as e:
            raise ImageProcessingError("Failed to detect image orientation", str(e))

    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Correct image orientation to upright.

        Args:
            image: Image as numpy array.

        Returns:
            Upright image as numpy array.

        Raises:
            ImageProcessingError: If orientation correction fails.
        """
        rotation = self.detect_orientation(image)

        if rotation == 0:
            return image

        if rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        logger.warning(f"Unexpected rotation angle: {rotation}, returning original image")
        return image

    def select_text_regions(
        self, image: np.ndarray
    ) -> List[np.ndarray]:
        """Identify text-containing regions in image.

        Uses OpenCV contours to find likely label areas on bottles.

        Args:
            image: Image as numpy array (should be upright).

        Returns:
            List of cropped text regions as numpy arrays.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter for label-like regions (rectangular, large area)
            regions = []
            h, w = image.shape[:2]
            min_area = 0.05 * h * w  # At least 5% of image area
            max_area = 0.7 * h * w  # No more than 70% of image area

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    # Get bounding rect
                    x, y, cw, ch = cv2.boundingRect(cnt)

                    # Aspect ratio check (labels are usually wider than tall)
                    aspect_ratio = float(cw) / ch if ch > 0 else 0
                    if aspect_ratio > 0.5 and aspect_ratio < 5.0:
                        # Crop region
                        region = image[y : y + ch, x : x + cw]
                        regions.append(region)

            # If no good regions found, return full image as single region
            if not regions:
                logger.warning("No suitable text regions found, using full image")
                return [image]

            # Sort regions by area (largest first)
            regions.sort(key=lambda r: r.shape[0] * r.shape[1], reverse=True)

            # Return top 5 regions at most
            return regions[:5]

        except Exception as e:
            logger.warning(f"Failed to select text regions, using full image: {e}")
            return [image]

    def extract_with_retry(
        self,
        image_path: Path,
        orientation_corrected: bool = False,
        select_regions: bool = True,
    ) -> str:
        """Extract text with automatic retry on failure.

        Wraps extract_text() with retry_manager for resilience.

        Args:
            image_path: Path to image file.
            orientation_corrected: Whether image is already upright.
            select_regions: Whether to select text regions before OCR.

        Returns:
            Extracted text as string.

        Raises:
            RetryExhaustedError: If all retry attempts exhausted.
            FileNotFoundError: If image file not found.
            ImageProcessingError: If image cannot be loaded.
        """
        return self.retry_manager.retry(
            self.extract_text,
            image_path,
            orientation_corrected=orientation_corrected,
            select_regions=select_regions,
        )

    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on PIL Image.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as string.

        Raises:
            OCRError: If OCR fails.
        """
        try:
            # Tesseract configuration:
            # psm=6: Assumes a single uniform block of text
            text = pytesseract.image_to_string(
                image,
                config="--psm 6",
            )
            return text
        except Exception as e:
            raise OCRError("OCR text extraction failed", str(e))
