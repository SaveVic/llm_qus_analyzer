from dataclasses import dataclass
from typing import Any, Optional
from ..utils import analyze_individual_with_basic, analyze_individual_with_llm
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation

_definition = """
A user story should express only one feature. 

"""

_in_format = """
Given user story "{user_story}" and its [Means] "{means}".
Without adding any context outside this user story, 
Evaluate and extract the individual, unique tasks explicitly mentioned strictly from the [Means] section.

- A task is indicated only by an explicit verb or a verb phrase applied to an object or objects.
- Extract tasks only when a verb is present. (Nouns alone are not tasks.)
- Do NOT treat adjectives or states (e.g., “relevant”, “available”, “visible”, “easy”) as tasks, even if listed alongside verbs.
- If a phrase contains a verb + adjective, extract only the verb action; treat the adjective as a qualifier of that task and do not output it as a separate task.
- Do NOT infer or invent actions not explicitly stated in the [Means].

"""

_in_format_no_chunk = """
Given user story "{user_story}"
**Evaluate whether this user story is 'Atomic' based on its [Role] and [Means]:**
- [Role]: A stakeholder or persona that expresses the need. Typically, [Role] are taken from the softwares application domain.
- [Means]: The phrase or clause that describes the primary capability, action, or system behavior the [Role] wants to perform or see happen. It represents the core functional need of the user story, including any conditions (triggers or preconditions) that are directly tied to that need, indicating by 'when', 'where', etc.

** [Role] check **
- Does the [Role]  only contain  1 actors?
** [Means] check **
- Does the [Means] only contain a single action that represent a singular task 
    A signular task is indicated only by explicit verb or  a verb phaase applied to an object(s)
    Do not treat adjective or states (e.g.  )



"""

_out_format = """
**Strictly follow this output format (JSON):**  
```json
{{  
    "tasks": "List of string"
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class MeansTasksData:
    """Container for parsed task information extracted from a Means component."""

    tasks: list[str]
    """List of discrete tasks identified in the Means component. Empty list if no valid tasks were found."""


class MeansTasksParserModel:
    """Model for parsing and analyzing task information from user story Means components.

    Uses an LLM analyzer to identify discrete tasks within Means components.
    """

    def __init__(self):
        """Initializes the parser model with LLM analyzer configuration."""
        self.key = "means-tasks"
        self.__analyzer = LLMAnalyzer[MeansTasksData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> MeansTasksData:
        """Parses raw JSON output into structured MeansTasksData.

        Args:
            raw_json (Any): The raw JSON output from the LLM analyzer.

        Returns:
            MeansTasksData: Structured representation of parsed tasks.

        Note:
            Handles various input formats by:
            - Converting string 'none' or empty strings to empty list
            - Wrapping single string tasks in a list
        """
        if not isinstance(raw_json, dict):
            return MeansTasksData([])
        tasks = raw_json.get("tasks", [])
        if isinstance(tasks, str):
            if tasks.lower() == "none" or tasks == "":
                tasks = []
            else:
                tasks = [tasks]
        return MeansTasksData(tasks=tasks)

    def analyze_single(
        self, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[str], LLMUsage | None]:
        """Analyzes a single user story component for discrete tasks.

        Args:
            client (LLMClient): Configured LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component (QUSComponent): Parsed user story components to analyze.

        Returns:
            tuple[list[str],Optional[LLMUsage]]:
                - List of identified tasks (empty if no Means component exists)
                - LLM usage metrics if analysis was performed, None otherwise
        """
        if component.means is None:
            return [], None
        values = {"user_story": component.text, "means": component.means}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.tasks, usage

    def analyze_list(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[str], LLMUsage | None]]:
        """Batch analyzes multiple user story components for discrete tasks.

        Args:
            client (LLMClient): Configured LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of parsed user story components to analyze.

        Returns:
            list[tuple[list[str],Optional[LLMUsage]]]:
                List of analysis results (tasks, usage) for each input component
        """
        return [
            self.analyze_single(client, model_idx, component)
            for component in components
        ]


class AtomicAnalyzer:
    """Analyzer for enforcing atomicity in user stories.

    Validates that user stories contain:
    - Exactly one Role
    - Means components representing a single atomic task
    """

    __mt_parser = MeansTasksParserModel()  # Shared task parser instance

    @classmethod
    def __is_role_single(cls, component: QUSComponent) -> Optional[Violation]:
        """Validates that the user story has exactly one Role.

        Args:
            component (QUSComponent): Parsed user story components to validate.

        Returns:
            Optional[Violation]:
                - Violation if multiple Roles exist
                - None if zero or one Role exists
        """
        role = component.role
        if not role:
            return None

        if len(role) > 1:
            tmp = ", ".join(role)
            return Violation(
                parts=set(["role"]),
                issue=f"The [Role] is more than 1: {tmp}",
                suggestion="Select one [Role] that suitable to the user story, or separate it to the different user story.",
            )

        return None

    @classmethod
    def __is_means_single_task(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[Optional[Violation], Optional[LLMUsage]]:
        """Validates that Means represents a single atomic task using LLM analysis.

        Args:
            client (LLMClient): Configured LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component (QUSComponent): Parsed user story components to validate.

        Returns:
            tuple[Optional[Violation],Optional[LLMUsage]]:
                - Violation if multiple tasks are identified
                - LLM usage metrics from the task analysis
        """
        means = component.means
        if not means:
            return None, None

        tasks, result = cls.__mt_parser.analyze_single(client, model_idx, component)

        if len(tasks) > 1:
            tmp = "\n".join([f"({i+1}) {task}" for i, task in enumerate(tasks)])
            return (
                Violation(
                    parts=set(["means"]),
                    issue=f"The [Means] contain more than 1 tasks:\n{tmp}",
                    suggestion="Select one task in [Means] that suitable to the user story, or separate it to the different user story.",
                ),
                result,
            )

        return None, result

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Executes atomicity validation checks on a user story.

        Args:
            client (LLMClient): Configured LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component (QUSComponent): Parsed user story components to validate.

        Returns:
            tuple[list[Violation],dict[str,LLMUsage]]:
                - List of all atomicity violations found
                - Dictionary of LLM usage metrics by analysis type
        """
        basic_checker = [cls.__is_role_single]
        violations = analyze_individual_with_basic(basic_checker, component)

        llm_checker = [cls.__is_means_single_task]
        task_keys = [cls.__mt_parser.key]
        more_violations, usages = analyze_individual_with_llm(
            llm_checker, client, model_idx, component
        )
        llm_usage = {k: r for k, r in zip(task_keys, usages) if r is not None}

        violations.extend(more_violations)

        return violations, llm_usage
