from typing import Optional
from ..client import LLMClient, LLMUsage
from ..chunker.models import QUSComponent
from ..utils import analyze_individual_with_basic
from ..type import Violation


class WellFormAnalyzer:
    """Analyzer for checking basic well-formedness of user story components.

    This class provides static methods to validate that user stories contain
    the required components (Role and Means) according to quality user story guidelines.
    """

    @classmethod
    def __does_have_role(cls, component: QUSComponent) -> Optional[Violation]:
        """Checks if the user story contains a valid Role component.

        Args:
            component (QUSComponent): The parsed user story components to validate.

        Returns:
            Optional[Violation]:
                - Violation object if Role is missing
                - None if Role is present and valid

        Note:
            A valid Role should be a non-empty list of strings representing
            stakeholder personas (e.g., ["user"], ["admin"], ["customer"]).
        """
        role = component.role
        if not role or len(role) == 0:
            return Violation(
                parts=set(["role"]),
                issue="The [Role] is missing",
                suggestion='Add the [Role] related to the user story, such as "user", etc.',
            )
        return None

    @classmethod
    def __does_have_means(cls, component: QUSComponent) -> Optional[Violation]:
        """Checks if the user story contains a valid Means component.

        Args:
            component (QUSComponent): The parsed user story components to validate.

        Returns:
            Optional[Violation]:
                - Violation object if Means is missing
                - None if Means is present and valid

        Note:
            Means represents the core action or capability in the user story.
            Even stories without Roles should typically have Means.
        """
        means = component.means
        if not means:
            return Violation(
                parts=set(["means"]),
                issue="The [Means] is missing",
                suggestion="Add the [Means] related to the user story.",
            )
        return None

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Runs all well-formedness checks on a user story component.

        Args:
            client (LLMClient): LLM client (currently unused, maintained for interface consistency)
            model_idx (int): Model index (currently unused, maintained for interface consistency)
            component (QUSComponent): Parsed user story components to validate

        Returns:
            tuple[list[Violation],dict[str, LLMUsage]]:
                - List of found violations (empty if story is well-formed)
                - Empty dictionary (maintained for interface consistency)

        Note:
            Current checks include:
            1. Presence of Role component
            2. Presence of Means component

            The LLM client parameter is not currently used but maintained for
            future extensibility and interface consistency with other analyzers.
        """
        basic_checker = [
            cls.__does_have_role,
            cls.__does_have_means,
        ]

        violations = analyze_individual_with_basic(basic_checker, component)

        return violations, {}
