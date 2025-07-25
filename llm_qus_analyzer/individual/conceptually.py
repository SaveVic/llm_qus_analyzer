from dataclasses import dataclass
from typing import Any, Optional
from ..utils import analyze_individual_with_llm
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation


_definition = """
**Evaluate whether this user story is 'Conceptually Sound' based on its [Means] and [Ends]:**  
1. **[Means] Check:**  
   - Is it a **single, concrete action** the system can perform directly?  
   - Does it **avoid hidden dependencies** (e.g., assuming unstated features)?  

2. **[Ends] Check:**  
   - Does it explain the **user's true goal or benefit** (not a system feature or intermediate step)?  
   - Is it **independent** (no implied dependencies)? 
"""

_in_format = """
**User Story to Evaluate:**  
- [Means]: {means}
- [Ends]: {ends}
"""

_out_format = """
**Stricly follow this output format (JSON) without any other explanation:**  
- If valid: `{{ "valid": true }}`  
- If invalid:  
  ```json
  {{
      "valid": false,
      "violations": [
        {{
            "part": "[Means]" or "[Ends]",
            "issue": "Description of the flaw",
            "suggestion": "How to fix it"
        }}
      ]
  }}
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class CSVerdictData:
    """Data class representing the verdict of a conceptual soundness analysis."""

    valid: bool
    """Boolean indicating whether the component is conceptually sound."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


_PART_MAP = {
    "[Means]": "means",
    "[Ends]": "ends",
}


class CSVerdictParserModel:
    """Parser model for analyzing conceptual soundness of QUS components using LLM.

    This class handles the parsing and analysis of QUS components to determine
    if they are conceptually sound according to the defined criteria.
    """

    def __init__(self):
        """Initializes the parser model with analyzer configuration."""
        self.key = "conceptually-sound"
        self.__analyzer = LLMAnalyzer[CSVerdictData](key=self.key)
        self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> CSVerdictData:
        """Parses raw JSON output from LLM into structured CSVerdictData.

        Args:
            raw_json: Raw JSON output from the LLM analysis.

        Returns:
            CSVerdictData: Containing the parsed validation results and violations.
        """
        if not isinstance(raw_json, dict):
            return CSVerdictData(False, [])

        valid = raw_json.get("valid", False)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = False

        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown", "Unknown")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    part = _PART_MAP.get(t.get("part", ""))
                    violations.append(
                        Violation(
                            parts={part} if part else {},
                            issue=t.get("issue", ""),
                            suggestion=t.get("suggestion"),
                        )
                    )
        if not valid and len(violations) == 0:
            violations.append(default_vio)

        return CSVerdictData(valid=valid, violations=violations)

    def analyze_single(
        self, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], LLMResult | None]:
        """Analyzes a single QUS component for conceptual soundness.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.

        Returns:
            Tuple containing list of violations and LLM result/usage data.
        """
        if component.means is None:
            return [], None
        values = {"means": component.means, "ends": component.ends}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.violations, usage

    def analyze_list(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> list[tuple[list[str], LLMResult | None]]:
        """Analyzes a list of QUS components for conceptual soundness.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            components (QUSComponent): List of QUSComponents to analyze.

        Returns:
            List of tuples containing violations and LLM results for each component.
        """
        return [
            self.analyze_single(client, model_idx, component)
            for component in components
        ]


class ConceptuallySoundAnalyzer:
    """Main analyzer class for conceptual soundness evaluation.

    Provides class methods for running conceptual soundness checks on QUS components.
    """

    __cs_parser = CSVerdictParserModel()

    @classmethod
    def __not_violated(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], Optional[LLMUsage]]:
        """Checks if a component violates conceptual soundness rules.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.

        Returns:
            Tuple containing list of violations and LLM usage data.
        """
        means = component.means
        if not means:
            return [], None

        violations, result = cls.__cs_parser.analyze_single(
            client, component, model_idx
        )
        return violations, result

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Runs the complete conceptual soundness analysis pipeline.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component (QUSComponent): QUSComponent to analyze.

        Returns:
            Tuple containing:
            - List of all violations found
            - Dictionary of LLM usage statistics by task key
        """
        llm_checker = [cls.__not_violated]
        task_keys = [cls.__cs_parser.key]
        violations, usages = analyze_individual_with_llm(
            llm_checker, client, model_idx, component
        )
        llm_usage = {k: r for k, r in zip(task_keys, usages) if r is not None}

        return violations, llm_usage
