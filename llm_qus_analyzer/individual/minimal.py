from typing import Optional, Any
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..utils import analyze_individual_with_basic, analyze_individual_with_llm
from ..type import Violation
from dataclasses import dataclass
import re
 
from ..analyzer import LLMAnalyzer

_desc="""
Does the special character within parentheses refer to an abbreviation.
"""
_in_format = """
Given a user story: "{text}"
And the content within parentheses: "{parentheses_content}"

Determine if the content within parentheses is an abbreviation of an object in this  user story 
Such as , "Model Context Protocol (MCP)" "Application Programming Interface (API)"
"""
_out_format = """
**Strictly follow this output format (JSON):**
```json
{{
    "is_abbreviation": true/false
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
""" 

@dataclass
class AbbreviationVerdictData:
    is_abbreviation: bool
    """Boolean indicating whether the content within parentheses is an abbreviation."""


class AbbreviationParserModel:
    """Model for parsing and analyzing abbreviations within parentheses using LLM.
    
    Uses an LLM analyzer to determine if content within parentheses represents an abbreviation.
    """
    def __init__(self):
        """Initializes the parser model with LLM analyzer configuration."""
        self.key = "abbreviation-check"
        self.__analyzer = LLMAnalyzer[AbbreviationVerdictData](key=self.key)
        self.__analyzer.build_prompt(_desc, _in_format, _out_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))
    
    def __parser(self, raw_json: Any) -> AbbreviationVerdictData:
        """Parses raw JSON output into structured AbbreviationVerdictData.
        
        Args:
            raw_json (Any): The raw JSON output from the LLM analyzer.
            
        Returns:
            AbbreviationVerdictData: Structured representation of abbreviation verdict.
        """
        if not isinstance(raw_json, dict):
            return AbbreviationVerdictData(False)
        is_abbrev = raw_json.get("is_abbreviation", False)
        return AbbreviationVerdictData(is_abbreviation=is_abbrev)
    
    def analyze_single(
        self, client: LLMClient, model_idx: int, text: str, parentheses_content: str
    ) -> tuple[bool, LLMUsage | None]:
        """Analyzes if content within parentheses is an abbreviation.
        
        Args:
            client (LLMClient): Configured LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            text (str): Full user story text.
            parentheses_content (str): Content within parentheses to analyze.
            
        Returns:
            tuple[bool, Optional[LLMUsage]]:
                - Boolean indicating if content is an abbreviation
                - LLM usage metrics if analysis was performed, None otherwise
        """
        values = {"text": text, "parentheses_content": parentheses_content}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.is_abbreviation, usage

class MinimalAnalyzer:
    """Analyzer for checking basic syntax and structure of user stories.

    Performs minimal validation checks to ensure user stories:
    - Contain only allowed special characters
    - Don't have unnecessary trailing information
    """

    @classmethod
    def __is_not_contain_special(cls, component: QUSComponent) -> Optional[Violation]:
        """Checks if the user story contains invalid special characters.

        Args:
            component (QUSComponent): The parsed user story component to validate

        Returns:
            Optional[Violation]:
                - Violation if invalid characters are found
                - None if only allowed characters are present

        Note:
            Allowed characters include:
            - Alphanumeric characters (a-z, A-Z, 0-9)
            - Basic punctuation (apostrophes, commas, periods, hyphens)
            - Spaces
        """
        us = component.text
        def not_valid(ch: str) -> bool:
            """Helper function to identify invalid characters.
            Args:
                ch (str): Single character to check

            Returns:
                bool: True if character is invalid, False otherwise
            """
            if ch.isalnum():
                return False
            if ch in ["'", ",", ".", "-", " ", "/"]:
                return False
            return True

        chars = [ch for ch in us if not_valid(ch)]
        chars = list(set(chars))  # Get unique invalid characters

        if len(chars) > 0:
            tmp = ", ".join(chars)
            return Violation(
                parts=set(["full"]),
                issue=f'The user story contains invalid characters: "{tmp}"',
                suggestion="Remove invalid characters from the user story",
            )

        return None
    @staticmethod
    def _check_parentheses(text: str) -> list[str]:
        """Extracts content within parentheses from text.
        Args:
            text (str): Text to search for parentheses content
        Returns:
            list[str]: List of content found within parentheses
        """
        pattern = r'\(([^)]+)\)'
        matches = re.findall(pattern, text)
        return matches
    
    @staticmethod
    def _has_parentheses(text: str) -> bool:
        """Checks if text contains parentheses.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text contains parentheses, False otherwise
        """
        return '(' in text and ')' in text
    __abbrev_parser = AbbreviationParserModel()  # Shared abbreviation parser instance
    
    @classmethod
    def __is_the_parentheses_an_abbreviation(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[Optional[Violation], Optional[LLMUsage]]:
        """Checks if parentheses contain abbreviations (false positives for special chars).
        
        Args:
            client (LLMClient): Configured LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component (QUSComponent): The parsed user story component to validate
            
        Returns:
            tuple[Optional[Violation], Optional[LLMUsage]]:
                - Violation if parentheses contain non-abbreviation content
                - LLM usage metrics from analysis
                
        Note:
            If parentheses contain abbreviations, this is considered a false positive
            and no violation is reported.
        """
        text = component.text
        if not cls._has_parentheses(text):
            return None, None
        parentheses_contents = cls._check_parentheses(text)
        if not parentheses_contents:
            return None, None
        total_usage = None
        for content in parentheses_contents:
            is_abbrev, usage = cls.__abbrev_parser.analyze_single(
                client, model_idx, text, content
            )
            if usage and total_usage:
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
            elif usage:
                total_usage = usage
            if not is_abbrev:
                return Violation(
                    parts=set(["full"]),
                    issue=f'The user story contains special characters in parentheses that are not abbreviations: "({content})"',
                    suggestion="Remove non-abbreviation content from parentheses or use abbreviations instead"
                ), total_usage
        return None, total_usage
    @classmethod
    def __is_us_no_tail(cls, component: QUSComponent) -> Optional[Violation]:
        """Checks if the user story has unnecessary trailing information.

        Args:
            component (QUSComponent): The parsed user story component to validate

        Returns:
            Optional[Violation]:
                - Violation if trailing information is found
                - None if no trailing information exists

        Note:
            Trailing information refers to text after the main user story
            components that doesn't contribute to the core functionality.
        """
        tail = component.template.tail

        if tail:
            return Violation(
                parts=set(["full"]),
                issue=f'The user story contains unnecessary info: "{tail}"',
                suggestion="Remove unnecessary info from the user story.",
            )

        return None

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, component: QUSComponent
    ) -> tuple[list[Violation], dict[str, LLMUsage]]:
        """Executes all minimal validation checks on a user story.

        Args:
            client (LLMClient): LLM client for LLM-based analysis
            model_idx (int): Model index for LLM analysis
            component (QUSComponent): Parsed user story components to validate

        Returns:
            tuple[list[Violation],dict[str,LLMUsage]]:
                - List of found violations (empty if all checks pass)
                - Dictionary of LLM usage metrics by analysis type

        Note:
            Current checks include:
            1. Valid character check
            2. Trailing information check
            3. Abbreviation false positive check for parentheses
        """
        basic_checker = [cls.__is_us_no_tail]
        violations = analyze_individual_with_basic(basic_checker, component)
        
        # Check for special characters
        special_violation = cls.__is_not_contain_special(component)
        if special_violation and cls._has_parentheses(component.text):
            # Only check for abbreviations if we have parentheses
            abbrev_violation, abbrev_usage = cls.__is_the_parentheses_an_abbreviation(
                client, model_idx, component
            )
            
            # If it's NOT an abbreviation, add the special character violation
            if abbrev_violation is not None:
                violations.append(special_violation)
            # If it IS an abbreviation, don't add the violation (false positive)
            
            usage_dict = {cls.__abbrev_parser.key: abbrev_usage} if abbrev_usage else {}
            return violations, usage_dict
        elif special_violation:
            # Special characters but no parentheses - regular violation
            violations.append(special_violation)
        
        return violations, {}
