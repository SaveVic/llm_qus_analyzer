from copy import deepcopy
from typing import Any, Optional
from ..type import Violation
from ..client import LLMClient, LLMUsage
from ..chunker.models import QUSComponent
from ..chunker.parser import Template


class UniformAnalyzer:
    """Analyzer for enforcing template uniformity across user stories.

    Identifies the most common template pattern among user stories and flags
    stories that deviate significantly from this pattern.
    """

    __PUNCT = '[PUNCT]'
    """Special token for punctuation marks"""

    __PUNCT_WEIGHT = 0.2
    """Reduced weight for punctuation in distance calculations"""

    __THRESHOLD = 3
    """Maximum allowed template deviation score"""

    @classmethod
    def __pos_distance(cls, list1: list[str], list2: list[str]) -> float:
        """Calculates weighted edit distance between two POS tag sequences.

        Args:
            list1 (list[str]): First sequence of POS tags
            list2 (list[str]): Second sequence of POS tags

        Returns:
            float: Weighted edit distance where punctuation has reduced impact

        Note:
            Uses dynamic programming with custom weights:
            - Regular tags have weight 1.0
            - Punctuation tags have weight __PUNCT_WEIGHT (0.2)
        """
        m, n = len(list1), len(list2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Initialize first row and column
        for i in range(1, m + 1):
            weight = cls.__PUNCT_WEIGHT if list1[i - 1] == cls.__PUNCT else 1
            dp[i][0] = dp[i - 1][0] + weight

        for j in range(1, n + 1):
            weight = cls.__PUNCT_WEIGHT if list2[j - 1] == cls.__PUNCT else 1
            dp[0][j] = dp[0][j - 1] + weight

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                w_i = cls.__PUNCT_WEIGHT if list1[i - 1] == cls.__PUNCT else 1
                w_j = cls.__PUNCT_WEIGHT if list2[j - 1] == cls.__PUNCT else 1
                candidates = [
                    dp[i - 1][j] + w_i,  # Deletion
                    dp[i][j - 1] + w_j,  # Insertion
                    dp[i - 1][j - 1] + w_i + w_j,  # Substitution
                ]
                if list1[i - 1] == list2[j - 1]:  # Match
                    candidates.append(dp[i - 1][j - 1])
                dp[i][j] = min(candidates)

        return dp[m][n]

    @classmethod
    def __existence_handling(cls, temp1: Template, temp2: Template, token: str, optional: bool = False) -> float:
        """Calculates distance penalty for missing components.

        Args:
            temp1 (Template): First template to compare
            temp2 (Template): Second template to compare
            token (str): Component token to check ('[ROLE]', '[MEANS]', '[ENDS]')
            optional (bool): Whether the component is optional (default: False)

        Returns:
            float: Distance penalty based on proper separation:
                   - ROLE and MEANS are mandatory and must exist separately
                   - ENDS is optional but when present should be in separate template
        """
        dist = 0.0
        sep1 = temp1.chunk.get(token, [])
        sep2 = temp2.chunk.get(token, [])
        
        # Ensure proper component separation in templates
        if token == '[ENDS]':
            # ENDS is optional but should be consistently present or absent
            if len(sep1) == 0 and len(sep2) == 0:
                dist = 0.0  # Both empty is fine
            elif len(sep1) == 0 or len(sep2) == 0:
                # One has ENDS, other doesn't - minor penalty for inconsistency
                dist = 0.5
            else:
                # Both have ENDS, compare template patterns
                dist = cls.__pos_distance(sep1, sep2)
        else:
            # ROLE and MEANS must be present and properly separated
            if len(sep1) == 0 or len(sep2) == 0:
                # Missing mandatory component gets full penalty
                dist = 2.0
            else:
                # Compare template patterns for proper separation
                dist = cls.__pos_distance(sep1, sep2)
            
        # Clean up for subsequent comparisons
        if token in temp1.chunk:
            temp1.chunk.pop(token, None)
        if token in temp2.chunk:
            temp2.chunk.pop(token, None)
        temp1.order = [t for t in temp1.order if t != token]
        temp2.order = [t for t in temp2.order if t != token]
        return dist

    @classmethod
    def __get_index(cls, seq: list, value: Any) -> Optional[int]:
        """Finds first index of value in sequence.

        Args:
            seq (list): Sequence to search
            value (Any): Value to find

        Returns:
            int: Index if found, None otherwise
        """
        for i, v in enumerate(seq):
            if v == value:
                return i
        return None

    @classmethod
    def __order_handling(cls, temp1: Template, temp2: Template, token: str, optional: bool = False) -> float:
        """Calculates distance penalty for component order mismatches with proper separation.

        Args:
            temp1 (Template): First template to compare
            temp2 (Template): Second template to compare
            token (str): Component token to check
            optional (bool): Whether the component is optional

        Returns:
            float: Combined distance from POS difference and order mismatch,
                   emphasizing proper role/means/ends template separation
        """
        idx1 = cls.__get_index(temp1.order, token)
        idx2 = cls.__get_index(temp2.order, token)
        chunk1 = temp1.chunk.get(token, [])
        chunk2 = temp2.chunk.get(token, [])
        dist = 0.0
        # Ensure components exist separately in templates
        if token == '[ENDS]':
            # ENDS is optional but should be consistently handled
            if len(chunk1) == 0 and len(chunk2) == 0:
                dist = 0.0  # Both templates don't have ENDS
            elif len(chunk1) == 0 or len(chunk2) == 0:
                # One template has ENDS, other doesn't - consistency penalty
                dist = 0.3
            else:
                # Both have ENDS templates, compare patterns
                dist = cls.__pos_distance(chunk1, chunk2)
        else:
            # ROLE and MEANS must exist separately in templates
            if len(chunk1) == 0 or len(chunk2) == 0:
                # Missing component in template structure
                dist = 3.0  # Heavy penalty for missing mandatory components
            else:
                # Compare template patterns for proper separation
                dist = cls.__pos_distance(chunk1, chunk2)
        # Enforce proper component order in templates (ROLE -> MEANS -> ENDS)
        if idx1 is not None and idx2 is not None:
            if idx1 != idx2:
                # Order mismatch penalty - templates should have consistent structure
                w1 = sum([(cls.__PUNCT_WEIGHT if c == cls.__PUNCT else 1) for c in chunk1])
                w2 = sum([(cls.__PUNCT_WEIGHT if c == cls.__PUNCT else 1) for c in chunk2])
                dist += max(w1, w2) * 0.8  # Order consistency penalty
            # Ensure proper separation: ROLE should come before MEANS, MEANS before ENDS
            if token == '[ROLE]' and idx1 != 0:
                dist += 1.0  # ROLE should be first in template
            elif token == '[MEANS]' and idx1 <= 0:
                dist += 1.0  # MEANS should come after ROLE
            elif token == '[ENDS]' and idx1 <= 1:
                dist += 0.5  # ENDS should come after MEANS (if present)

        return dist

    @classmethod
    def __template_distance(cls, temp1: Template, temp2: Template) -> float:
        """Calculates comprehensive distance between two templates.

        Args:
            temp1 (Template): First template to compare
            temp2 (Template): Second template to compare

        Returns:
            float: Total distance score considering:
                   - Tail content
                   - Component existence
                   - Component order
                   - POS patterns
        """
        total_distance = 0.0
        _temp1 = deepcopy(temp1)
        _temp2 = deepcopy(temp2)
        # Penalize tail mismatch
        if temp1.tail or temp2.tail:
            total_distance += 1
        # Check component existence
        total_distance += cls.__existence_handling(_temp1, _temp2, '[ROLE]')
        total_distance += cls.__existence_handling(_temp1, _temp2, '[MEANS]')
        total_distance += cls.__existence_handling(
            _temp1, _temp2, '[ENDS]', optional=True)

        # Check component order and patterns
        total_distance += cls.__order_handling(_temp1, _temp2, '[ROLE]')
        total_distance += cls.__order_handling(_temp1, _temp2, '[MEANS]')
        total_distance += cls.__order_handling(
            _temp1, _temp2, '[ENDS]', optional=True)

        return total_distance

    @classmethod
    def __find_most_common_separators(cls, templates: list[Template]) -> tuple[list[str], list[str], list[str]]:
        """Finds most common role, means, and ends separators following chunker logic.
        
        Args:
            templates (list[Template]): List of templates to analyze
            
        Returns:
            tuple: (most_common_role_seps, most_common_means_seps, most_common_ends_seps)
        """
        from collections import Counter
        role_counter = Counter()
        means_counter = Counter()
        ends_counter = Counter()
        for template in templates:
            # Extract role separators
            role_chunk = template.chunk.get('[ROLE]', [])
            if role_chunk:
                role_sep = ' '.join(role_chunk).lower()
                role_counter[role_sep] += 1
                
            # Extract means separators  
            means_chunk = template.chunk.get('[MEANS]', [])
            if means_chunk:
                means_sep = ' '.join(means_chunk).lower()
                means_counter[means_sep] += 1
                
            # Extract ends separators
            ends_chunk = template.chunk.get('[ENDS]', [])
            if ends_chunk:
                ends_sep = ' '.join(ends_chunk).lower()
                ends_counter[ends_sep] += 1
        
        # Get most common separators
        most_common_role = role_counter.most_common(1)[0][0] if role_counter else ''
        most_common_means = means_counter.most_common(1)[0][0] if means_counter else ''
        
        # For ENDS, find most common non-empty separator (like original logic)
        most_common_ends = ''
        for sep, _ in ends_counter.most_common():
            if sep.strip():  # Skip empty separators
                most_common_ends = sep
                break
                
        return [most_common_role], [most_common_means], [most_common_ends]

    @classmethod
    def __find_most_common_template_text(cls, templates: list[Template]) -> str:
        """Finds the most common template text structure.
        
        Args:
            templates (list[Template]): List of templates to analyze
            
        Returns:
            str: Most common template text pattern
        """
        from collections import Counter
        template_counter = Counter()
        
        for template in templates:
            template_counter[template.text] += 1
        
        if template_counter:
            return template_counter.most_common(1)[0][0]
        return ""

    @classmethod
    def __generate_violation_data(cls, text_template: str, component: QUSComponent) -> Violation:
        """Generates violation details for template deviations with proper component separation.

        Args:
            text_template (str): The expected template pattern
            component (QUSComponent): The deviating user story component

        Returns:
            Violation: Structured suggestion for template alignment with separated components
        """
        # Extract components ensuring proper separation
        role_part = ' or '.join(component.role) if component.role else '[ROLE_MISSING]'
        means_part = component.means if component.means else '[MEANS_MISSING]'
        ends_part = component.ends if component.ends else '[ENDS_OPTIONAL]'
        
        # Create template suggestion with proper component separation
        template_parts = []
        if '[ROLE]' in text_template:
            template_parts.append(f"As a {role_part}")
        if '[MEANS]' in text_template:
            template_parts.append(f"I want {means_part}")
        if '[ENDS]' in text_template and ends_part != '[ENDS_OPTIONAL]':
            template_parts.append(f"so that {ends_part}")
        
        suggested_format = ', '.join(template_parts)
        
        # Identify specific issues with component separation
        issues = []
        if not component.role:
            issues.append("missing ROLE component")
        if not component.means:
            issues.append("missing MEANS component")
        
        issue_description = "Template structure issues: " + ", ".join(issues) if issues else f"User story doesn't follow the common template pattern: \"{text_template}\""
        
        return Violation(
            parts={'template', 'role', 'means', 'ends'},
            issue=issue_description,
            suggestion=f'Consider reformatting with proper component separation: "{suggested_format}"'
        )

    @classmethod
    def run(cls, client: LLMClient, model_idx: int, components: list[QUSComponent]) -> list[tuple[list[Violation], dict[str, LLMUsage]]]:
        """Analyzes user stories for template uniformity following original chunker logic.

        Args:
            client (LLMClient): LLM client (unused, maintained for interface consistency)
            model_idx (int): Model index (unused, maintained for interface consistency)
            components (list[QUSComponent]): List of user story components to analyze

        Returns:
            List of (violations, usage) tuples for each input component

        Note:
            - Finds most common ROLE and MEANS separators (collected together)
            - ENDS is handled separately (optional, allows empty)
            - Follows original chunker _uniform() logic
        """
        templates = [comp.template for comp in components]
        if not templates:
            return [([], {}) for _ in components]
        common_role_seps, common_means_seps, common_ends_seps = cls.__find_most_common_separators(templates)
        most_common_template_text = cls.__find_most_common_template_text(templates)
        
        results = []
        for comp in components:
            template = comp.template
            violations = []
            
            # Extract current component's separators
            role_chunk = template.chunk.get('[ROLE]', [])
            means_chunk = template.chunk.get('[MEANS]', [])
            ends_chunk = template.chunk.get('[ENDS]', [])
            
            current_role_sep = ' '.join(role_chunk).lower() if role_chunk else ''
            current_means_sep = ' '.join(means_chunk).lower() if means_chunk else ''
            current_ends_sep = ' '.join(ends_chunk).lower() if ends_chunk else ''
            
            # Check uniformity following original logic
            role_ok = False
            means_ok = False
            ends_ok = True  # ENDS is optional by default
            
            # ROLE check: flexible matching (startswith both ways)
            if common_role_seps and common_role_seps[0]:
                expected_role = common_role_seps[0]
                role_ok = (current_role_sep.startswith(expected_role) or 
                          expected_role.startswith(current_role_sep))
            
            # MEANS check: strict prefix matching
            if common_means_seps and common_means_seps[0]:
                expected_means = common_means_seps[0]
                means_ok = current_means_sep.startswith(expected_means)
            
            # ENDS check: empty is allowed, otherwise must match
            if current_ends_sep == '':
                ends_ok = True  # Empty ends is always okay
            elif common_ends_seps and common_ends_seps[0]:
                expected_ends = common_ends_seps[0]
                ends_ok = current_ends_sep.startswith(expected_ends)
            
            # Generate violation if any component doesn't match
            if not (role_ok and means_ok and ends_ok):
                # Use actual template text, handle optional ENDS
                expected_template = most_common_template_text
                
                # If current story has no ENDS but template includes ENDS, strip ENDS part
                if not comp.ends and "{ENDS}" in expected_template:
                    # Remove ENDS and trailing connectors (so that, in order to, etc.)
                    ends_patterns = [" so that {ENDS}", " in order to {ENDS}", " to {ENDS}", ", {ENDS}"]
                    for pattern in ends_patterns:
                        if pattern in expected_template:
                            expected_template = expected_template.replace(pattern, "")
                            break
                
                violation = cls.__generate_violation_data(expected_template, comp)
                violations.append(violation)
            
            results.append((violations, {}))
        
        return results
