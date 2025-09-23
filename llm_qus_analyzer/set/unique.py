import re
from collections import defaultdict
from ..analyzer import LLMAnalyzer
from ..client import LLMClient, LLMResult, LLMUsage
from ..chunker.models import QUSComponent
from ..type import Violation, PairwiseViolation, FullSetViolation
from ..utils import analyze_set_pairwise, analyze_set_fullset, format_set_results_pairwise, format_set_results_fullset
from dataclasses import dataclass
from typing import Any, Optional

# Optional dependencies for semantic analysis
try:
    import stanza
    import spacy
    HAS_NLP_DEPS = True
except ImportError:
    stanza = None
    spacy = None
    HAS_NLP_DEPS = False

_definition = """
**Evaluate whether two user stories are 'Semantically Similar' despite different wording:**
1. **Similar Action Check:**
   - Do both stories describe essentially the same functionality ?
   - Are the core behaviors or system operations equivalent ?
   
2. **Similar Output Check:**
   - Do both stories aim for the same outcome or benefit?
   - Are the end goals or results essentially the same?

Note: Stories are semantically similar if they request the same thing using different words.
**Suggestions to fix:**
 - Concatenate the story into one 
 
"""

_in_format = """
**User Stories to Compare:**
id: {id1} story: "{story1}"
id: {id2} story: "{story2}"
"""

_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If not similar: `{{"valid": true}}`
- If semantically similar:
  ```json
  {{
      "valid": false,
      "violations": [
        {{
            "id_pair": {{"first": 0, "second": 1}},
            "issue": "Description of how the stories are semantically similar",
            "first_suggestion": "How to consolidate or differentiate the first story",
            "second_suggestion": "How to consolidate or differentiate the second story"
        }}
      ]
  }}
  ```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_all_set_definition = """
**Evaluate whether multiple user stories are 'Unique' by checking for various duplicate relationships across the entire set:**

1. **Full Duplicate Check:**
   - Are there any stories that are textually identical?
   - Full duplicates have identical text content.

2. **Semantic Duplicate Check:**
   - Are there stories that request the same thing using different words?
   - Different text but same functional requirement.

3. **Different Means, Same End Check:**
   - Do any stories have the same goal/outcome but achieve it using different methods?
   - Same end result, different approaches or actions.

4. **Same Means, Different End Check:**
   - Do any stories use the same action/method to reach different goals?
   - Same functionality serving different purposes.

5. **Different Role, Same Means/End Check:**
   - Do any stories have different roles but same functionality or outcomes?
   - Different users requesting similar features.

6. **Purpose = Means Check:**
   - Is the end/goal of one story identical to the action/means of another story?
   - One story's reason becomes another story's action.

Note: Stories violate uniqueness if they have any of these duplicate relationships. Analyze the Role, Means, and Ends components of each story to detect these patterns.
"""

_all_set_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_all_set_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If unique: `{{"valid": true}}`
- If duplicates found:
```json
{{
    "valid": false,
    "violations": [
      {{
          "story_ids": [1, 3, 5],
          "parts_per_story": [["text"], ["text"], ["text"]],
          "issue": "Description of the duplicate stories found",
          "suggestion": "How to resolve the duplicate stories in this set"
      }}
    ]
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""

_dependency_definition = """
**Evaluate whether multiple user stories are 'Semantically Similar' despite different wording:**

1. **Similar Functionality Check:**
   - Do these stories describe essentially the same feature or capability?
   - Are the core behaviors or system operations equivalent?
   
2. **Similar Outcome Check:**
   - Do these stories aim for the same result or benefit?
   - Are the end goals essentially the same?

3. **Role-Means-Ends Analysis:**
   - Compare the Role (who), Means (how), and Ends (what/why) components
   - Stories are duplicates if they have similar role-means-ends combinations

Note: Stories are semantically similar if they request the same thing using different words or phrasing.
**Suggestions to fix:**
 - Merge semantically similar stories into one comprehensive story
 - Remove duplicate stories that don't add unique value
"""

_dependency_in_format = """
**User Stories to Evaluate:**
{stories_list}
"""

_dependency_out_format = """
**Strictly follow this output format (JSON) without any other explanation:**
- If no conflicts: `{{"valid": true}}`
- If dependency conflicts found:
```json
{{
    "valid": false,
    "violations": [
      {{
          "story_ids": [1, 3, 5],
          "parts_per_story": [["means"], ["means"], ["means"]],
          "issue": "Description of the dependency conflict found",
          "suggestion": "How to resolve the conflicting actions in this set"
      }}
    ]
}}
```
**Please only display the final answer without any explanation, description, or any redundant text.**
"""


@dataclass
class UniqueVerdictData:
    """Data class representing the verdict of a uniqueness analysis."""

    valid: bool
    """Boolean indicating whether the components are unique (not duplicates)."""

    violations: list[Violation]
    """List of Violation objects found in the analysis."""


@dataclass
class UniqueFullSetVerdictData:
    """Data class representing the verdict of a full-set uniqueness analysis."""

    valid: bool
    """Boolean indicating whether the components are unique (not duplicates)."""

    violations: list[FullSetViolation]
    """List of FullSetViolation objects found in the analysis."""


def format_stories_list(components: list[QUSComponent]) -> str:
    """Formats a list of QUSComponent objects into a numbered story list for LLM input.

    Args:
        components: List of QUSComponent objects to format

    Returns:
        Formatted string with numbered stories
    """
    return "\n".join([
        f"id: {comp.id or f'story_{i+1}'} story: \"{comp.text}\""
        for i, comp in enumerate(components)
    ])


_PART_MAP = {
    "semantic_duplicate": "semantic",
}


class UniqueParserModel:
    """Unified parser for uniqueness analysis supporting both pairwise and fullset modes."""

    def __init__(self, mode: str):
        """Initialize parser with specified mode.

        Args:
            mode: Either "pairwise", "fullset", or "dependency"
        """
        if mode not in ["pairwise", "fullset", "dependency"]:
            raise ValueError("Mode must be 'pairwise', 'fullset', or 'dependency'")

        self.mode = mode
        self.key = f"unique-{mode}"

        if mode == "pairwise":
            self.__analyzer = LLMAnalyzer[UniqueVerdictData](key=self.key)
            self.__analyzer.build_prompt(_definition, _in_format, _out_format)
        elif mode == "fullset":
            self.__analyzer = LLMAnalyzer[UniqueFullSetVerdictData](key=self.key)
            self.__analyzer.build_prompt(_all_set_definition, _all_set_in_format, _all_set_out_format)
        else:  # dependency
            self.__analyzer = LLMAnalyzer[UniqueFullSetVerdictData](key=self.key)
            self.__analyzer.build_prompt(_dependency_definition, _dependency_in_format, _dependency_out_format)

        self.__analyzer.build_parser(lambda raw: self.__parser(raw))

    def __parser(self, raw_json: Any) -> UniqueVerdictData | UniqueFullSetVerdictData:
        """Parses raw JSON output from LLM into structured data.
        Args:
            raw_json: Raw JSON output from the LLM analysis.
        Returns:
            UniqueVerdictData or UniqueFullSetVerdictData depending on mode.
        """
        if not isinstance(raw_json, dict):
            if self.mode == "pairwise":
                return UniqueVerdictData(True, [])
            else:
                return UniqueFullSetVerdictData(True, [])

        valid = raw_json.get("valid", True)
        if isinstance(valid, str):
            valid = valid == "true"
        elif valid is None:
            valid = True

        if self.mode == "pairwise":
            return self.__parse_pairwise(raw_json, valid)
        else:
            return self.__parse_fullset(raw_json, valid)

    def __parse_pairwise(self, raw_json: dict, valid: bool) -> UniqueVerdictData:
        """Parse pairwise analysis result."""
        violations: list[Violation] = []
        default_vio = Violation({}, "Unknown semantic similarity", "Review stories for duplicates")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    # Extract id_pair for uniqueness pair identification
                    id_pair = t.get("id_pair", {})
                    first_idx = id_pair.get("first", 0) if isinstance(id_pair, dict) else 0
                    second_idx = id_pair.get("second", 1) if isinstance(id_pair, dict) else 1

                    # Default parts for uniqueness analysis
                    first_parts = {"text"}
                    second_parts = {"text"}

                    # Store both parts in violation for later PairwiseViolation creation
                    violation = Violation(
                        parts=first_parts.union(second_parts),
                        issue=t.get("issue", ""),
                        suggestion=t.get("first_suggestion", t.get("suggestion", "")),
                    )
                    # Store additional data for PairwiseViolation
                    violation._first_parts = first_parts
                    violation._second_parts = second_parts
                    violation._second_suggestion = t.get("second_suggestion", t.get("suggestion", ""))
                    violation._id_pair = {"first": first_idx, "second": second_idx}

                    violations.append(violation)
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return UniqueVerdictData(valid=valid, violations=violations)

    def __parse_fullset(self, raw_json: dict, valid: bool) -> UniqueFullSetVerdictData:
        """Parse fullset analysis result."""
        violations: list[FullSetViolation] = []
        default_vio = FullSetViolation([], [], "Unknown duplicates", "Review stories for uniqueness")
        tmp = raw_json.get("violations", [])
        if isinstance(tmp, list):
            for t in tmp:
                if isinstance(t, dict):
                    story_ids = t.get("story_ids", [])
                    if isinstance(story_ids, list):
                        story_ids = [int(sid) - 1 for sid in story_ids if isinstance(sid,
                                                                                     (int, str)) and str(sid).isdigit()]
                    else:
                        story_ids = []

                    parts_per_story = t.get("parts_per_story", [])
                    if not isinstance(parts_per_story, list):
                        parts_per_story = []

                    # Convert string parts to sets
                    processed_parts = []
                    for parts in parts_per_story:
                        if isinstance(parts, list):
                            part_set = set()
                            for part in parts:
                                # For uniqueness, usually just "text"
                                part_set.add(part.lower())
                            processed_parts.append(part_set)
                        else:
                            processed_parts.append({"text"})

                    violations.append(
                        FullSetViolation(
                            story_ids=story_ids,
                            parts_per_story=processed_parts,
                            issue=t.get("issue", ""),
                            suggestion=t.get("suggestion", "")
                        )
                    )
        if not valid and len(violations) == 0:
            violations.append(default_vio)
        return UniqueFullSetVerdictData(valid=valid, violations=violations)

    def analyze_pairwise(
        self, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], LLMResult | None]:
        """Analyzes two QUS components for semantic similarity.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component1 (QUSComponent): First QUSComponent to compare.
            component2 (QUSComponent): Second QUSComponent to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM result.
        """
        if self.mode != "pairwise":
            raise ValueError("This parser is not in pairwise mode")

        values = {
            "id1": component1.id,
            "id2": component2.id,
            "story1": component1.text,
            "story2": component2.text,
        }
        data, usage = self.__analyzer.run(client, model_idx, values)

        pairwise_violations: list[PairwiseViolation] = []
        for violation in data.violations:
            # Use stored parts from parser if available, otherwise default to text
            first_parts = getattr(violation, '_first_parts', {"text"})
            second_parts = getattr(violation, '_second_parts', {"text"})
            second_suggestion = getattr(violation, '_second_suggestion', violation.suggestion)

            # Ensure we have proper suggestion format
            if second_suggestion and second_suggestion != violation.suggestion:
                combined_suggestion = f"First story: {violation.suggestion}. Second story: {second_suggestion}"
            else:
                combined_suggestion = violation.suggestion

            # Use component IDs if available, otherwise use placeholder values
            first_id = component1.id or "component_1"
            second_id = component2.id or "component_2"

            pairwise_violations.append(
                PairwiseViolation(
                    first_parts=first_parts,
                    second_parts=second_parts,
                    first_id=first_id,
                    second_id=second_id,
                    issue=violation.issue,
                    suggestion=combined_suggestion,
                )
            )
        return pairwise_violations, usage

    def analyze_full_set(
        self, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], LLMResult | None]:
        """Analyzes multiple QUS components for duplicates in a single LLM call.
        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            components (list[QUSComponent]): List of QUSComponents to analyze.
        Returns:
            Tuple containing list of full-set violations and LLM result.
        """
        if self.mode not in ["fullset", "dependency"]:
            raise ValueError("This parser is not in fullset or dependency mode")

        if len(components) < 2:
            return [], None

        stories_list = format_stories_list(components)
        values = {"stories_list": stories_list}
        data, usage = self.__analyzer.run(client, model_idx, values)
        return data.violations, usage


class UniqueAnalyzer:
    """Main analyzer class for uniqueness evaluation.

    Provides class methods for running uniqueness checks on sets of QUS components
    using both full duplicate detection and semantic similarity analysis.
    """

    __unique_parser_pairwise = UniqueParserModel("pairwise")
    __unique_parser_fullset = UniqueParserModel("fullset")
    __unique_parser_dependency = UniqueParserModel("dependency")
    
    # NLP pipeline instances (lazy-loaded)
    _stanza_nlp = None
    _spacy_nlp = None
    _SIMILARITY_THRESHOLD = 0.6

    @classmethod
    def _init_nlp_pipelines(cls):
        """Initialize NLP pipelines for semantic analysis (lazy-loaded)."""
        if not HAS_NLP_DEPS:
            raise ImportError("stanza and spacy are required for semantic analysis")
        
        if cls._stanza_nlp is None:
            stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
            cls._stanza_nlp = stanza.Pipeline(
                "en", 
                processors="tokenize,pos,lemma,depparse",
                tokenize_pretokenized=False,
                verbose=False,
            )
        
        if cls._spacy_nlp is None:
            try:
                cls._spacy_nlp = spacy.load("en_core_web_md")
            except OSError:
                print("SpaCy model not found, downloading en_core_web_md...")
                spacy.cli.download("en_core_web_md")
                cls._spacy_nlp = spacy.load("en_core_web_md")

    @classmethod
    def _extract_action_and_objects(cls, means_text: str) -> tuple[Optional[str], set[str]]:
        """Extract verb and objects from means text using dependency parsing.
        
        Args:
            means_text: The means component text to analyze
            
        Returns:
            Tuple of (main_verb, set_of_objects)
        """
        if not means_text or not HAS_NLP_DEPS:
            return None, set()
            
        if cls._stanza_nlp is None:
            cls._init_nlp_pipelines()
            
        doc = cls._stanza_nlp(means_text)
        verb = None
        objects = set()
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Find root verb (head == 0)
                if word.head == 0:
                    verb = word.lemma.lower()
                # Find objects with specific dependency relations
                if word.deprel in {"obj", "iobj", "obl"} and word.upos in {"NOUN", "PROPN"}:
                    objects.add(word.lemma.lower())
                    
        return verb, objects

    @classmethod
    def _are_words_related(cls, word1: str, word2: str) -> bool:
        """Check if two words are semantically related using SpaCy similarity.
        
        Args:
            word1: First word to compare
            word2: Second word to compare
            
        Returns:
            True if words are semantically similar above threshold
        """
        if not HAS_NLP_DEPS or word1 == word2:
            return word1 == word2
            
        if cls._spacy_nlp is None:
            cls._init_nlp_pipelines()
            
        token1 = cls._spacy_nlp(word1)
        token2 = cls._spacy_nlp(word2)
        
        if not token1.has_vector or not token2.has_vector:
            return False
            
        similarity = token1.similarity(token2)
        return similarity > cls._SIMILARITY_THRESHOLD

    @classmethod
    def _is_full_duplicate(cls, component1: QUSComponent, component2: QUSComponent) -> bool:
        """Checks if two components are full duplicates using case-insensitive text comparison.
        Args:
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.
        Returns:
            bool: True if components are full duplicates, False otherwise.
        """
        if not component1.text or not component2.text:
            return False
        text1 = re.sub(r'\s+', ' ', component1.text.strip().lower())
        text2 = re.sub(r'\s+', ' ', component2.text.strip().lower())
        return text1 == text2

    @classmethod
    def _is_semantically_similar(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[Optional[PairwiseViolation], Optional[LLMUsage]]:
        """Checks if two components are semantically similar using LLM analysis.

        Args:
            client (LLMClient): LLMClient instance for making API calls.
            model_idx (int): Index of the LLM model to use for analysis.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing optional PairwiseViolation and LLM usage data.
        """
        violations, usage = cls.__unique_parser_pairwise.analyze_pairwise(
            client, model_idx, component1, component2
        )

        if violations:
            return violations[0], usage  # Return first violation

        return None, usage

    @classmethod
    def analyze_pairwise(
        cls, client: LLMClient, model_idx: int, component1: QUSComponent, component2: QUSComponent
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes two components for uniqueness violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            component1 (QUSComponent): First component to compare.
            component2 (QUSComponent): Second component to compare.

        Returns:
            Tuple containing list of pairwise violations and LLM usage data.
        """
        violations: list[PairwiseViolation] = []
        usage_dict: dict[str, LLMUsage] = {}

        # Step 1: Check for full duplicates (fast regex check)
        if cls._is_full_duplicate(component1, component2):
            # Use component IDs if available, otherwise use placeholder values
            first_id = component1.id or "component_1"
            second_id = component2.id or "component_2"

            violations.append(
                PairwiseViolation(
                    first_parts={"text"},
                    second_parts={"text"},
                    first_id=first_id,
                    second_id=second_id,
                    issue="Stories are identical duplicates",
                    suggestion="Remove one of the duplicate stories",
                )
            )
            return violations, usage_dict

        # Step 2: Check for semantic similarity (LLM analysis)
        semantic_violation, usage = cls._is_semantically_similar(
            client, model_idx, component1, component2
        )

        if semantic_violation:
            violations.append(semantic_violation)

        if usage:
            usage_dict[cls.__unique_parser_pairwise.key] = usage

        return violations, usage_dict

    @classmethod
    def analyze_all_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[PairwiseViolation], dict[str, LLMUsage]]:
        """Analyzes all pairwise combinations in a set for uniqueness violations.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of all pairwise violations and LLM usage data.
        """
        return analyze_set_pairwise(cls, client, model_idx, components)

    @classmethod
    def analyze_full_set(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], dict[str, LLMUsage]]:
        """Analyzes all components using FULLSET LLM mode: Collect → Format → LLM.
        
        Uses _all_set_definition to find duplicates across entire set.
        LLM returns story IDs that need to be mapped back to components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            components (list[QUSComponent]): List of components to analyze.

        Returns:
            Tuple containing list of full-set violations and LLM usage data.
        """
        if len(components) < 2:
            return [], {}

        # Use the fullset parser with _all_set_definition
        violations, usage = cls.__unique_parser_fullset.analyze_full_set(client, model_idx, components)
        
        # Convert usage to dict format
        usage_dict = {}
        if usage:
            usage_dict[cls.__unique_parser_fullset.key] = usage
            
        return violations, usage_dict

    @classmethod
    def analyze_set_dependency(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], dict[str, LLMUsage]]:
        """Analyzes components for semantic similarity using LLM.
        
        Uses simple clustering and LLM to find semantically similar stories.
        This is much simpler than complex verb-object analysis.
        
        Args:
            client: LLM client for analysis
            model_idx: Model index to use
            components: List of QUSComponent objects to analyze
            
        Returns:
            Tuple of (violations, usage_dict)
        """
        if len(components) < 2:
            return [], {}
        
        # Simple clustering by text similarity - just group similar length stories for now
        # This is much simpler than complex NLP analysis
        clusters = []
        remaining = components.copy()
        
        while remaining:
            current = remaining.pop(0)
            cluster = [current]
            
            # Group stories with similar length (simple heuristic)
            to_remove = []
            for other in remaining:
                if abs(len(current.text) - len(other.text)) < 50:  # Similar length
                    cluster.append(other)
                    to_remove.append(other)
            
            for item in to_remove:
                remaining.remove(item)
                
            if len(cluster) > 1:  # Only analyze clusters with multiple stories
                clusters.append(cluster)
        
        # Use LLM to analyze each cluster for semantic similarity
        violations = []
        all_usage = {}
        
        for cluster_idx, cluster in enumerate(clusters):
            # Use the dependency parser to analyze this cluster
            cluster_violations, usage = cls.__unique_parser_dependency.analyze_full_set(
                client, model_idx, cluster
            )
            
            violations.extend(cluster_violations)
            
            if usage:
                key = f"{cls.__unique_parser_dependency.key}_cluster_{cluster_idx}"
                all_usage[key] = usage
        
        return violations, all_usage

    @classmethod
    def analyze_dependency_llm(
        cls, client: LLMClient, model_idx: int, components: list[QUSComponent]
    ) -> tuple[list[FullSetViolation], dict[str, LLMUsage]]:
        """Analyzes components for dependency conflicts - CLUSTER FIRST, then batch LLM.
        
        Logic:
        1. Cluster similar dependencies using SpaCy/NLP
        2. Format each cluster into a list  
        3. Walk through LLM PER similar cluster (not per story)
        
        Args:
            client: LLM client for analysis
            model_idx: Index of the LLM model to use
            components: List of QUSComponent objects to analyze
            
        Returns:
            Tuple of (violations, usage_dict)
        """
        if len(components) < 2:
            return [], {}

        # First cluster using SpaCy-based similarity (reuse the clustering logic)
        clustered_violations, _ = cls.analyze_set_dependency(components)
        
        if not clustered_violations:
            return [], {}
        
        # Now batch each cluster through LLM for better analysis
        all_violations = []
        usage_dict = {}
        
        for i, cluster_violation in enumerate(clustered_violations):
            # Get components for this cluster and track original indices
            original_indices = cluster_violation.story_ids
            cluster_components = [components[idx] for idx in original_indices]
            
            # Run LLM on this cluster - input as formatted lists
            llm_violations, usage = cls.__unique_parser_dependency.analyze_full_set(
                client, model_idx, cluster_components
            )
            
            # Map returned story_ids back to original component indices
            for violation in llm_violations:
                # LLM returns 0-based indices within the cluster
                # Map them back to original component indices
                mapped_story_ids = [original_indices[cluster_idx] for cluster_idx in violation.story_ids 
                                  if cluster_idx < len(original_indices)]
                violation.story_ids = mapped_story_ids
            
            all_violations.extend(llm_violations)
            
            if usage:
                usage_dict[f"{cls.__unique_parser_dependency.key}_cluster_{i}"] = usage
            
        return all_violations, usage_dict

    @classmethod
    def run(
        cls, client: LLMClient, model_idx: int, *args, mode: str = "pairwise"
    ) -> tuple[list[PairwiseViolation | FullSetViolation], dict[str, LLMUsage]]:
        """Runs uniqueness analysis on user story components.

        Args:
            client (LLMClient): LLM client for analysis.
            model_idx (int): Index of the LLM model to use.
            *args: Variable arguments based on mode:
                - For pairwise mode: component1, component2 (two QUSComponent objects)
                - For fullset mode: components (list[QUSComponent])
                - For dependency mode: components (list[QUSComponent]) - uses LLM analysis
            mode (str): Analysis mode - "pairwise", "fullset", or "dependency". Defaults to "pairwise".

        Returns:
            Tuple containing violations and LLM usage data.

        Note:
            - Pairwise mode: Compares two individual components with duplicate and semantic analysis
            - Fullset mode: Analyzes entire set using O(n) hash-based duplicate detection
            - Dependency mode: Analyzes set using LLM for verb-object conflict detection with proper ID formatting
        """
        if mode == "pairwise":
            if len(args) != 2:
                raise ValueError("Pairwise mode requires exactly 2 components")
            component1, component2 = args
            return cls.analyze_pairwise(client, model_idx, component1, component2)

        elif mode == "fullset":
            if len(args) != 1 or not isinstance(args[0], list):
                raise ValueError("Fullset mode requires a list of components")
            components = args[0]
            if len(components) < 2:
                return [], {}
            return cls.analyze_full_set(client, model_idx, components)

        elif mode == "dependency":
            if len(args) != 1 or not isinstance(args[0], list):
                raise ValueError("Dependency mode requires a list of components")
            components = args[0]
            if len(components) < 2:
                return [], {}
            return cls.analyze_set_dependency(client, model_idx, components)

        else:
            raise ValueError("Mode must be 'pairwise', 'fullset', or 'dependency'")
