from dataclasses import dataclass
from typing import Any, Optional
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMUsage
from .parser import Template, TemplateParser

input_format = """
Expand every English verb contraction into its full form without changing tense, aspect, or meaning. 
Only replace the contracted token with its literal expansion, and do not modify any other words. For example, 
“I’m” becomes “I am” and “don’t” becomes “do not”. Words that are not contractions must remain exactly the same, 
so “I want” stays “I want” and “I would” stays “I would”.
"{user_story}"
**Please only display the single final answer without any explanation, fixing steps, or any redundant text.**
"""

output_format = """
Your response must be a single, valid JSON object. 
Do not include any text or formatting outside of the JSON structure.
The JSON object should strictly conform to the following schema:
- `expanded`: A string represent the expanded text of user story.
- `component`: An object containing:
  - `[Role]`: An array of string consist of all of the [Role] in the user story. If no [Role], then this should be an empty array `[]`.
  - `[Means]`: A string represent the [Means] of the user story. If no [Means], then this should be `null`.
  - `[Ends]`: A string represent the [Ends] of the user story. If no [Ends], then this should be `null`.
"""

cb_l = """
Based on the Quality User Story (QUS) framework, a user story consists of three parts:
[Role], [Means], and optionally [End]. By definition:
- [Role]: A stakeholder or persona that expresses the need. Typically, [Role] are taken from the softwares application domain.
- [Means]: The phrase or clause that describes the primary capability, action, or system behavior the [Role] wants to perform or see happen. It represents the core functional need of the user story, including any conditions (triggers or preconditions) that are directly tied to that need, indicating by 'when', 'where', etc.
- [Ends]: A direct value of the [Means] for the [Role] or explaining why the [Means] are requested or a dependency on other functionality of the [Means].
Additionally,
the [Role] can be the name of the persona acts as the role: Joe, Alice, and not a subject like I, you, they, etc.
the [Means] should not start with a phrasal modal verb (or semi-modal verb), like "be able to", "want to" etc. Even though if [Role] is not exists, [Means] still can be exists.
the [Ends] should start with a pronoun (if it exist) or verb, not a causal phrase such as "so that", "in order to", "to" etc, and not including any unnecessary text behind it.
Every [Role], [Means] and [Ends] must be **explicitly mentioned** or become a part or substring of the user story.
"""

@dataclass
class QUSChunkData:
    """Container for the analyzed components of a user story."""
    expanded: str
    """The user story with all contractions expanded."""
    role: list[str]
    """List of identified role(s) in the user story."""
    means: Optional[str]
    """The identified means component, if present."""
    ends: Optional[str]
    """The identified ends component, if present."""

@dataclass
class QUSComponent:
    """Comprehensive representation of a parsed user story with template."""
    text: str
    """The expanded text of the user story."""
    
    role: list[str]
    """List of identified role(s)."""
    
    means: Optional[str]
    """The means component, if present."""
    
    ends: Optional[str]
    """The ends component, if present."""
    
    template: Template
    """Templatized version of the user story."""
    
    id: Optional[str] = None
    """Optional unique identifier for the component."""
    
    original_text: Optional[str] = None
    """The original user story text before expansion."""

class QUSChunkerModel:
    """Model for analyzing and chunking user stories into components.

    Uses an LLM analyzer to identify roles, means, and ends components,
    then creates templates from the analyzed stories.
    """
    def __init__(self) -> None:
        """Initializes the chunker model with predefined prompts and parser."""
        self.key = "chunker"
        self.__analyzer = LLMAnalyzer[QUSChunkData](key=self.key)
        self.__analyzer.build_prompt(cb_l, input_format, output_format)
        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw: Any) -> QUSChunkData:
        """Parses raw LLM output into structured type.
        Args:
            raw (Any): The raw JSON output from the LLM analyzer.
        Returns:
            QUSChunkData: Structured representation of the parsed components.
        Note:
            Handles various edge cases in the raw output including:
            - Normalizing role to always be a list
            - Converting string 'none' or empty strings to None
            - Handling both list and string representations of roles
        """
        expanded = raw["expanded"]
        role = raw["component"]["[Role]"]
        if role is None:
            role = []
        if isinstance(role, str):
            if role.lower() == "none" or role == "":
                role = []
            else:
                role = [role]
        means = raw["component"]["[Means]"]
        if isinstance(means, list):
            if not means:
                raise ValueError("LLM returned empty array for [Means]")
            means = means[0]
        elif isinstance(means, str):
            if means.lower() == "none" or means == "":
                means = None
        ends = raw["component"]["[Ends]"]
        if isinstance(ends, list):
            if not ends:
                print("[SNAFU]: LLM returned empty array for [Ends]")
                raise ValueError("LLM returned empty array for [Ends]")
            print(f"[SNAFU]: LLM returned array for [Ends]: {ends}, taking first element")
            ends = ends[0]
        elif isinstance(ends, str):
            if ends.lower() == "none" or ends == "":
                ends = None
        return QUSChunkData(expanded, role, means, ends)

    def analyze_single(
        self, client: LLMClient, model_idx: int, user_story: str, id: Optional[str] = None
    ) -> tuple[QUSComponent, LLMUsage]:
        """Analyzes a single user story into its components.

        Args:
            client (LLMClient): The LLM client to use for analysis.
            model_idx (int): Index of the specific LLM model to use.
            user_story (str): The user story text to analyze.
            id (Optional[str]): Optional unique identifier for the component.

        Returns:
            tuple[QUSComponent,LLMUsage]:
                - The fully parsed user story components
                - The LLM usage object
        Note:
            The analysis pipeline:
            1. LLM extracts roles, means, and ends
            2. TemplateParser creates a template pattern
            3. Results are packaged into QUSComponent
        """
        values = {"user_story": user_story}
        data, usage = self.__analyzer.run(client, model_idx, values)
        TemplateParser.prepare()
        template = TemplateParser.parse(data.expanded, data.role, data.means, data.ends)
        component = QUSComponent(
            text=data.expanded,
            role=data.role,
            means=data.means,
            ends=data.ends,
            template=template,
            id=id,
            original_text=user_story,
        )
        return component, usage

    def analyze_list(
        self, client: LLMClient, model_idx: int, user_stories: list[str], ids: list[str] = None
    ) -> list[tuple[QUSComponent, LLMUsage]]:
        """Analyzes multiple user stories in batch.

        Args:
            client: The LLM client to use for analysis.
            model_idx: Index of the specific LLM model to use.
            user_stories: List of user story texts to analyze.
            ids: Optional list of IDs corresponding to the user stories.

        Returns:
            list[tuple[QUSComponent,LLMUsage]]:
                List of analysis results (component, usage) for each input story.
        """
        if ids is None:
            ids = [None] * len(user_stories)
        elif len(ids) != len(user_stories):
            raise ValueError("Length of ids must match length of user_stories")

        return [
            self.analyze_single(client, model_idx, user_story, id)
            for user_story, id in zip(user_stories, ids)
        ]
