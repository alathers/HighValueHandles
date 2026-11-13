"""Interactive naming tool for reviewing and correcting bottle metadata.

Computes preliminary bottle name from distillery + release info + proof,
allows user review and corrections.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from models.bottle import Bottle, Correction
from lib.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class NameComponents:
    """Parsed components for bottle naming.

    Attributes:
        distillery: Distillery name.
        release: Specific release information (style/series/finish).
        proof: Proof statement.
        year: Age statement (optional).
    """

    distillery: str
    release: str
    proof: Optional[str] = None
    year: Optional[str] = None

    def __init__(self, distillery: str, release: str, proof: Optional[str] = None, year: Optional[str] = None):
        self.distillery = distillery.strip()
        self.release = release.strip()
        self.proof = proof.strip() if proof else None
        self.year = year.strip() if year else None


@dataclass
class ReviewResult:
    """Result of user review session.

    Attributes:
        approved: Whether user approved the metadata.
        corrections: List of corrections made by user.
        final_metadata: Final metadata after corrections.
    """

    approved: bool
    corrections: List[Correction] = field(default_factory=list)
    final_metadata: Optional[dict] = None


class NamingTool:
    """Interactive tool for reviewing and correcting bottle metadata.

    Computes preliminary name, presents preview, collects user corrections.
    """

    def __init__(self, interactive: bool = False):
        """Initialize naming tool.

        Args:
            interactive: Whether to use interactive prompts (vs. programmatic API).
        """
        self.interactive = interactive

    def compute_preliminary_name(self, components: NameComponents) -> str:
        """Compute preliminary bottle name from components.

        Format: "{distillery} {release} ({proof})"

        Args:
            components: Name components with distillery, release, proof.

        Returns:
            Preliminary bottle name string.

        Example:
            >>> components = NameComponents("Blanton's", "Single Barrel", "107 Proof")
            >>> compute_preliminary_name(components)
            "Blanton's Single Barrel (107 Proof)"
        """
        parts = [components.distillery]

        if components.release:
            parts.append(components.release)

        if components.proof:
            parts.append(f"({components.proof})")

        return " ".join(parts)

    def extract_components(self, metadata: dict) -> NameComponents:
        """Extract name components from bottle metadata.

        Args:
            metadata: Bottle metadata dict with distillery, specific_release,
                      proof/proof_statement, age/year fields.

        Returns:
            NameComponents instance.

        Raises:
            ValidationError: If required components missing.
        """
        # Required fields
        distillery = metadata.get("distillery")
        if not distillery:
            raise ValidationError("Missing required field: distillery")

        release = metadata.get("specific_release") or metadata.get("release")
        if not release:
            raise ValidationError("Missing required field: specific_release")

        # Optional fields
        proof = (
            metadata.get("proof")
            or metadata.get("proof_statement")
            or metadata.get("abv")
        )
        year = metadata.get("age") or metadata.get("year")

        return NameComponents(distillery=distillery, release=release, proof=proof, year=year)

    def format_preview(self, components: NameComponents, ocr_text: str, image_paths: List[Path]) -> str:
        """Format preview text for user review.

        Args:
            components: Parsed name components.
            ocr_text: OCR extracted text.
            image_paths: List of image paths.

        Returns:
            Formatted preview string.
        """
        lines = ["=" * 60]
        lines.append("🍸 Bottle Metadata Preview")
        lines.append("=" * 60)
        lines.append("")

        lines.append("Proposed Name:")
        name = self.compute_preliminary_name(components)
        lines.append(f"  {name}")
        lines.append("")

        lines.append("Components:")
        lines.append(f"  Distillery: {components.distillery}")
        lines.append(f"  Release:    {components.release}")

        if components.proof:
            lines.append(f"  Proof:      {components.proof}")

        if components.year:
            lines.append(f"  Age:        {components.year}")

        lines.append("")

        lines.append("OCR Preview:")
        lines.append("-" * 40)
        lines.append("```")
        preview = ocr_text[:200]
        if len(ocr_text) > 200:
            preview += "..."
        lines.append(preview)
        lines.append("```")
        lines.append("")

        lines.append("Images:")
        for i, path in enumerate(image_paths, 1):
            lines.append(f"  {i}. {path.name}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_user_corrections(
        self, components: NameComponents, metadata: dict
    ) -> List[Correction]:
        """Get user corrections via interactive prompt.

        Args:
            components: Current name components.
            metadata: Current bottle metadata dict.

        Returns:
            List of Correction objects from user.

        Note:
            This method uses interactive prompts and requires user input.
        """
        corrections = []

        print("\nReview each field:")
        print("Press Enter to accept, or type the corrected value.")

        # Review distillery
        current = components.distillery
        print(f"\n1. Distillery: {current}")
        user_input = input("Correction (or Enter to keep): ").strip()
        if user_input and user_input != current:
            corrections.append(
                Correction(field_name="distillery", original_value=current, corrected_value=user_input)
            )

        # Review release
        current = components.release
        print(f"\n2. Release: {current}")
        user_input = input("Correction (or Enter to keep): ").strip()
        if user_input and user_input != current:
            corrections.append(
                Correction(
                    field_name="specific_release", original_value=current, corrected_value=user_input
                )
            )

        # Review proof
        current = components.proof or "N/A"
        print(f"\n3. Proof: {current}")
        user_input = input("Correction (or Enter to keep): ").strip()
        if user_input and user_input != current:
            corrections.append(
                Correction(field_name="proof", original_value=current, corrected_value=user_input)
            )

        # Review age
        current = components.year or "N/A"
        print(f"\n4. Age: {current}")
        user_input = input("Correction (or Enter to keep): ").strip()
        if user_input and user_input != current:
            corrections.append(
                Correction(field_name="age", original_value=current, corrected_value=user_input)
            )

        # Review other metadata fields
        for key in ["msrp", "tasting_notes"]:
            current_val = metadata.get(key, "N/A")
            if key == "msrp" and current_val != "N/A":
                current_val = f"${current_val}"

            print(f"\n5. {key.title()}: {current_val}")
            user_input = input("Correction (or Enter to keep): ").strip()
            if user_input and user_input != current_val:
                # Strip $ symbol from price if present
                if key == "msrp" and user_input.startswith("$"):
                    user_input = user_input[1:]
                corrections.append(
                    Correction(field_name=key, original_value=str(current_val), corrected_value=user_input)
                )

        return corrections

    def apply_corrections(self, metadata: dict, corrections: List[Correction]) -> dict:
        """Apply corrections to metadata dict.

        Args:
            metadata: Original metadata dict.
            corrections: List of corrections to apply.

        Returns:
            Updated metadata dict.
        """
        updated = metadata.copy()

        for correction in corrections:
            field = correction.field_name
            updated[field] = correction.corrected_value
            logger.info(f"Applied correction: {field} = '{correction.corrected_value}'")

        return updated

    def create_bottle_from_metadata(self, metadata: dict) -> Bottle:
        """Create Bottle object from corrected metadata.

        Args:
            metadata: metadata dict with all bottle fields.

        Returns:
            Bottle object.
        """
        from ..models.bottle import ExtractedMetadata, Quantity

        bottle = Bottle(
            name=metadata.get("name", ""),
            distillery=metadata.get("distillery", ""),
            specific_release=metadata.get("specific_release", ""),
            msrp=metadata.get("msrp", "0.00"),
            proof=metadata.get("proof"),
            age=metadata.get("age"),
            tasting_notes=metadata.get("tasting_notes"),
            quantity=Quantity(
                count=metadata.get("quantity", 1),
                price=metadata.get("price", "0.00"),
                source=metadata.get("source", "extracted"),
            ),
        )

        return bottle

    def review_and_correct(
        self, metadata: dict, ocr_text: str, image_paths: List[Path]
    ) -> ReviewResult:
        """Interactive review and correction workflow.

        Args:
            metadata: Bottle metadata dict (preliminary).
            ocr_text: OCR extracted text.
            image_paths: List of image paths.

        Returns:
            ReviewResult with approval status and corrections.

        Workflow:
        1. Extract components from metadata
        2. Compute preliminary name
        3. Display preview
        4. Get user corrections (interactive)
        5. Apply corrections to metadata
        6. Return review result
        """
        # Extract components
        components = self.extract_components(metadata)

        # Compute preliminary name
        name = self.compute_preliminary_name(components)
        metadata["name"] = name  # Set preliminary name

        # Display preview
        preview = self.format_preview(components, ocr_text, image_paths)
        print(f"\n{preview}")

        # Get corrections
        corrections: List[Correction] = []
        if self.interactive:
            corrections = self.get_user_corrections(components, metadata)

            # Ask for approval
            print("\n" + "=" * 60)
            approve = input("Approve metadata? (y/n): ").strip().lower()

            approved = approve == "y"

            if approved and corrections:
                # Apply corrections
                metadata = self.apply_corrections(metadata, corrections)

                # Re-compute name with corrections
                updated_components = self.extract_components(metadata)
                metadata["name"] = self.compute_preliminary_name(updated_components)

                final_name = metadata["name"]
                print(f"\n✅ Approved with corrections. Final name: {final_name}")

                return ReviewResult(
                    approved=True, corrections=corrections, final_metadata=metadata
                )
            elif approved:
                print(f"\n✅ Approved. Name: {name}")
                return ReviewResult(approved=True, corrections=[], final_metadata=metadata)
            else:
                print("\n❌ Rejected by user.")
                return ReviewResult(approved=False, corrections=corrections, final_metadata=None)
        else:
            # Non-interactive mode
            return ReviewResult(
                approved=True, corrections=[], final_metadata=metadata
            )


def compute_name(
    distillery: str, release: str, proof: Optional[str] = None, year: Optional[str] = None
) -> str:
    """Convenience function to compute bottle name.

    Args:
        distillery: Distillery name.
        release: Release info.
        proof: Proof statement.
        year: Age statement.

    Returns:
        Formatted bottle name.
    """
    components = NameComponents(distillery=distillery, release=release, proof=proof, year=year)
    tool = NamingTool()
    return tool.compute_preliminary_name(components)
