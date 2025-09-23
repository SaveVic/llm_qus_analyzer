from copy import deepcopy
from dataclasses import dataclass
import stanza
from typing import Optional


@dataclass
class WordInfo:
    """Represents a single word with its linguistic properties and position."""

    text: str
    """The word text in lowercase."""

    start: int
    """Starting character position in the original text."""

    end: int
    """Ending character position in the original text."""

    pos: str
    """Part-of-speech tag in square brackets (e.g., '[NOUN]')."""

    @staticmethod
    def copy(other: "WordInfo") -> "WordInfo":
        """Creates a deep copy of a WordInfo instance.

        Args:
            other (WordInfo): The WordInfo instance to copy.

        Returns:
            WordInfo: A new WordInfo instance with identical attributes.
        """
        return WordInfo(other.text, other.start, other.end, other.pos)


@dataclass
class Template:
    """Represents a templatized user story with identified components."""

    text: str
    """The templatized text with components replaced by tokens."""

    chunk: dict[str, list[str]]
    """Dictionary mapping component tokens to their POS patterns."""

    tail: Optional[str]
    """Any remaining non-templatized text."""

    order: list[str]
    """Sequence of component tokens in the order they appear."""

    @staticmethod
    def copy(other: "Template") -> "Template":
        """Creates a deep copy of a Template instance.

        Args:
            other (Template): The Template instance to copy.

        Returns:
            Template: A new Template instance with identical attributes.
        """
        return Template(
            other.text,
            deepcopy(other.chunk),
            other.tail,
            other.order[:],
        )


class TemplateParser:
    """A utility class for extracting and templatizing patterns from user stories.

    This class provides methods to identify and replace common patterns (roles, means, ends)
    in user stories with standardized tokens, helping to create templates from concrete examples.
    """
    __reserve_pos = ["[ROLE]", "[MEANS]", "[ENDS]"]
    """Special POS tags for components"""
    __valid_chars = ["."]
    """Allowed non-alphanumeric characters"""

    @classmethod
    def prepare(cls) -> None:
        """Downloads and initializes the Stanza NLP processor.
        Note:
            This should be called once before any parsing operations.
        """
        if hasattr(cls, "_TemplateParser__posser"):
            return
        print("Downloading stanza processor")
        stanza.download("en", verbose=False)
        cls.__posser = stanza.Pipeline("en", processors="tokenize,pos", verbose=False)

    @classmethod
    def __contain_non_alnum(cls, text: str) -> bool:
        """Checks if text contains any non-alphanumeric characters.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if any non-alphanumeric character is found, False otherwise.
        """
        for c in text:
            if not c.isalnum():
                return True
        return False

    @classmethod
    def __tokenize(cls, text: str, non_alnum_ok: bool = True) -> list[WordInfo]:
        """Tokenizes and tags text with part-of-speech information.

        Args:
            text (str): The text to tokenize.
            non_alnum_ok (bool): Whether to include tokens with non-alphanumeric characters.

        Returns:
            list[WordInfo]: List of WordInfo objects representing each token.
        """
        doc = cls.__posser(text)
        words: list[WordInfo] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if not non_alnum_ok:
                    if cls.__contain_non_alnum(word.text):
                        continue
                pos = f"[{word.pos}]"
                words.append(
                    WordInfo(word.text.lower(), word.start_char, word.end_char, pos)
                )
        return words

    @classmethod
    def __lcs(
        cls, list1: list[WordInfo], list2: list[WordInfo], first_prio=True
    ) -> tuple[int, int]:
        """Finds the longest common subsequence between two token lists.

        Args:
            list1 (list[WordInfo]): First list of WordInfo tokens.
            list2 (list[WordInfo]): Second list of WordInfo tokens.
            first_prio (bool): When True, prioritizes earliest matches when scores are tied.
                               When False, prioritizes later matches. Defaults to True.

        Returns:
            tuple[int,int]: Tuple of (start_index, end_index) of the LCS in list1.

        Note:
            Uses dynamic programming with custom scoring that prioritizes:
            1. Match length
            2. Match span length
            3. Earlier matches
        """
        m, n = len(list1), len(list2)
        dp = [[(0, 0, -1, -1)] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                candidates = [dp[i - 1][j], dp[i][j - 1]]
                if list1[i - 1].text == list2[j - 1].text:
                    value = dp[i - 1][j - 1]
                    start = value[2]
                    length = (
                        list1[i - 1].end
                        - (list1[i - 1] if start < 0 else list1[start]).start
                    )
                    candidates.append(
                        (value[0] + 1, length, i - 1 if start < 0 else start, i - 1)
                    )
                candidates.sort(
                    key=lambda c: c[0] * 1e4
                    - c[1] * 1e2
                    + (-c[2] if first_prio else c[2]),
                    reverse=True,
                )
                dp[i][j] = candidates[0][:]

        return dp[m][n][2], dp[m][n][3]

    @classmethod
    def __refine_list(
        cls, words: list[WordInfo], token: str, start: int, end: int
    ) -> list[WordInfo]:
        """Replaces a subsequence of tokens with a component token.

        Args:
            words (list[WordInfo]): Original list of WordInfo tokens.
            token (str): Component type ('ROLE', 'MEANS', or 'ENDS').
            start (int): Start index of subsequence to replace.
            end (int): End index of subsequence to replace.

        Returns:
            list[WordInfo]: New list of WordInfo tokens with the subsequence replaced.
        """
        new_words: list[WordInfo] = []
        for i, word in enumerate(words):
            if i == start:
                pos = f"[{token}]"
                new_words.append(
                    WordInfo(f"{{{token}}}", words[start].start, words[end].end, pos)
                )
            elif i < start or i > end:
                new_words.append(word)
        return new_words

    @classmethod
    def __construct_text_words(cls, words: list[WordInfo], text: str) -> str:
        """Reconstructs text from WordInfo tokens while preserving original spacing.

        Args:
            words (list[WordInfo]): List of WordInfo tokens.
            text (str): Original text for reference.

        Returns:
            str: Reconstructed text string.
        """
        str_list: list[str] = []
        if len(words) > 0:
            str_list.append(text[words[0].start : words[0].end])
        for i in range(1, len(words)):
            spaces = " " * (words[i].start - words[i - 1].end)
            str_list.append(spaces)
            if words[i].pos not in cls.__reserve_pos:
                str_list.append(text[words[i].start : words[i].end])
            else:
                str_list.append(words[i].text)
        return "".join(str_list)

    @classmethod
    def __contain_invalid_char(cls, text: str) -> bool:
        """Checks if text contains any invalid characters.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if any invalid character is found, False otherwise.
        """
        for ch in text:
            if ch not in cls.__valid_chars and not ch.isalnum():
                return True
        return False

    @classmethod
    def __detect_tail_start(cls, words: list[WordInfo]) -> int:
        """Finds the starting index of non-templatized tail content.

        Args:
            words (list[WordInfo]): List of WordInfo tokens.

        Returns:
            int: Index where tail content begins.
        """
        last_token_idx = -1
        for i, word in enumerate(words):
            if word.pos in cls.__reserve_pos:
                last_token_idx = i
        tail_start_idx = last_token_idx + 1
        for i in range(last_token_idx + 1, len(words)):
            tail_start_idx = i
            if cls.__contain_invalid_char(words[i].text):
                break
            else:
                tail_start_idx += 1
        return tail_start_idx

    @classmethod
    def __construct_template(
        cls, words: list[WordInfo], text: str, tail: str
    ) -> Template:
        """Constructs a Template object from processed tokens.

        Args:
            words (list[WordInfo]): List of WordInfo tokens.
            text (str): Templatized text.
            tail (str): Non-templatized tail content.

        Returns:
            Template: A Template object representing the parsed structure.
        """
        chunk: dict[str, list[str]] = {}
        order: list[str] = []
        buffer: list[WordInfo] = []
        for word in words:
            if word.pos in cls.__reserve_pos:
                order.append(word.pos)
                chunk[word.pos] = deepcopy(buffer)
                buffer = []
            else:
                buffer.append(word.pos)
        if len(buffer) > 0:
            chunk["[APPENDIX]"] = deepcopy(buffer)
        return Template(text, chunk, tail, order)

    @classmethod
    def parse(
        cls, text: str, role: list[str], means: Optional[str], ends: Optional[str]
    ) -> Template:
        """Main method to parse and templatize a user story.

        Args:
            text (str): The user story text to parse.
            role (list[str]): List of role terms to identify.
            means (str | None): Means component text to identify.
            ends (str | None): Ends component text to identify.

        Returns:
            Template: A Template object representing the parsed structure.

        Raises:
            NotImplementedError: If the parser hasn't been initialized with prepare()

        Processing Steps:
            1. Tokenize and POS tag the input text
            2. Identify and replace Means component if provided
            3. Identify and replace Ends component if provided
            4. Identify and replace Role component if provided
            5. Detect and separate any tail content
            6. Construct and return the Template object

        Examples:
            >>> TemplateParser.prepare()  # Initialize first
            >>> story = "As a user, I want to login so I can access my account"
            >>> template = TemplateParser.parse(story, ["user"], "want to login", "can access my account")
            >>> print(template.text)
            "As a {ROLE}, I {MEANS} so I {ENDS}"
        """

        if not hasattr(cls, "_TemplateParser__posser"):
            raise NotImplementedError(
                "Posser not initialized yet. Please call `prepare` first."
            )

        text_words = cls.__tokenize(text)

        # Means
        if means:
            means_words = cls.__tokenize(means, False)
            start, end = cls.__lcs(text_words, means_words, first_prio=False)
            text_words = cls.__refine_list(text_words, "MEANS", start, end)

        # Ends
        if ends:
            ends_words = cls.__tokenize(ends, False)
            start, end = cls.__lcs(text_words, ends_words, first_prio=False)
            text_words = cls.__refine_list(text_words, "ENDS", start, end)

        # Role
        if len(role) > 0:
            role_str = " ".join(role)
            role_words = cls.__tokenize(role_str, True)
            start, end = cls.__lcs(text_words, role_words)
            text_words = cls.__refine_list(text_words, "ROLE", start, end)

        tail_idx = cls.__detect_tail_start(text_words)
        tail = cls.__construct_text_words(text_words[tail_idx:], text)
        tail = tail.strip()
        if len(tail) == 0:
            tail = None

        text_words = [
            WordInfo.copy(word) for i, word in enumerate(text_words) if i < tail_idx
        ]
        text_template = cls.__construct_text_words(text_words, text)
        template = cls.__construct_template(text_words, text_template, tail)

        return template


if __name__ == "__main__":
    text = "As a good person, I want to be able to help other people when they stuck. - see good person"
    role = ["good person"]
    means = "help other people when they stuck"
    ends = None

    TemplateParser.prepare()
    template = TemplateParser.parse(text, role, means, ends)
    print(template.text)
    print(template.chunk)
    print(template.tail)
    print(template.order)
