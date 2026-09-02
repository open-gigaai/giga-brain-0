import json
import re
import sys
import unicodedata

from lmms_eval.api.filter import Filter


def _balanced_boxed_contents(text):
    r"""Return complete or recoverable ``\boxed{...}`` contents in source order."""
    text = str(text)
    contents = []
    for match in re.finditer(r"\\boxed\s*\{", text, flags=re.IGNORECASE):
        start = match.end()
        depth = 1
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[start:idx])
                    break
        else:
            contents.append(text[start:])
    return contents


def _document_choices(doc):
    for key in ("choices", "answers", "answer_options", "options"):
        choices = doc.get(key)
        if choices is None:
            continue
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                continue
        if isinstance(choices, dict):
            return [(str(key).strip().upper(), str(value).strip()) for key, value in choices.items()]
        if isinstance(choices, (list, tuple)):
            return [(chr(ord("A") + idx), str(value).strip()) for idx, value in enumerate(choices)]
    return [(letter, "") for letter in "ABCD"]


def extract_robust_choice(text, doc):
    """Extract an unambiguous multiple-choice letter while preserving invalid output."""
    original = str(text).strip()
    choices = _document_choices(doc or {})
    valid_letters = {letter for letter, _ in choices if len(letter) == 1 and letter.isalpha()}

    def extract_candidate(candidate):
        candidate = str(candidate).strip()
        nested = _balanced_boxed_contents(candidate)
        if nested:
            nested_answer = extract_candidate(nested[-1])
            if nested_answer is not None:
                return nested_answer

        explicit = re.findall(
            r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|[:：])\s*"
            r"[\(\[\{#]?\s*([A-Z])\b",
            candidate,
            flags=re.IGNORECASE,
        )
        if explicit and explicit[-1].upper() in valid_letters:
            return explicit[-1].upper()

        conclusion = re.findall(
            r"(?:correct\s+(?:answer|option|choice|classification)|therefore|thus|hence)"
            r".*?\b(?:is|would\s+be)\s*[\(\[\{]?\s*([A-Z])\b",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if conclusion and conclusion[-1].upper() in valid_letters:
            return conclusion[-1].upper()

        for pattern in (
            r"^\s*[\(\[\{]\s*([A-Z])\s*[\)\]\}]",
            r"^\s*#?\s*([A-Z])(?=$|[.\)\]\}:,;\\])",
        ):
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match and match.group(1).upper() in valid_letters:
                return match.group(1).upper()

        option_matches = re.findall(r"\boption\s*([A-Z])\b", candidate, flags=re.IGNORECASE)
        if option_matches and option_matches[-1].upper() in valid_letters:
            return option_matches[-1].upper()

        normalized = candidate.strip().strip("`*_ ").lower()
        for letter, value in choices:
            if value and normalized == value.lower():
                return letter

        option_mentions = []
        for letter, value in choices:
            value = value.strip().lower()
            if not value:
                continue
            for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", normalized, flags=re.IGNORECASE):
                option_mentions.append((letter, match.start(), match.end()))
        visible_mentions = [
            mention
            for mention in option_mentions
            if not any(
                other[1] <= mention[1]
                and other[2] >= mention[2]
                and (other[2] - other[1]) > (mention[2] - mention[1])
                for other in option_mentions
            )
        ]
        mentioned_letters = {letter for letter, _, _ in visible_mentions}
        if len(mentioned_letters) == 1:
            return mentioned_letters.pop()

        number = re.fullmatch(r"#?\s*(-?\d+(?:\.\d+)?)", normalized)
        if number:
            for letter, value in choices:
                if value == number.group(1):
                    return letter
        return None

    boxed = _balanced_boxed_contents(original)
    for candidate in reversed(boxed):
        answer = extract_candidate(candidate)
        if answer is not None:
            return answer
    answer = extract_candidate(original)
    return answer if answer is not None else original


def extract_robust_yes_no(text):
    """Extract a leading or explicitly-final yes/no answer and remove punctuation."""
    original = str(text).strip()
    candidates = list(reversed(_balanced_boxed_contents(original))) + [original]
    for candidate in candidates:
        candidate = str(candidate).strip()
        explicit = re.findall(
            r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|[:：])\s*(yes|no)\b",
            candidate,
            flags=re.IGNORECASE,
        )
        if explicit:
            return explicit[-1].lower()
        match = re.match(r"^\s*(yes|no)\b", candidate, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return original


class RobustChoiceFilter(Filter):
    """Canonicalize recoverable choice answers to one uppercase letter."""

    def apply(self, resps, docs):
        return [
            extract_robust_choice(resp_set[0] if resp_set else "", doc)
            for resp_set, doc in zip(resps, docs)
        ]


class RobustYesNoFilter(Filter):
    """Canonicalize recoverable yes/no answers to lowercase text."""

    def apply(self, resps, docs):
        return [extract_robust_yes_no(resp_set[0] if resp_set else "") for resp_set in resps]


class WhitespaceFilter(Filter):
    """ """

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.startswith(" "):
                    resp = resp[1:]

                filtered_resp.append(resp)

            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


class RegexFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        # print(resps)
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # print(filtered_resps)

        return filtered_resps


class MultiChoiceRegexFilter(RegexFilter):
    """
    A filter used to extract a model's answer on multiple choice questions with
    letter answers. assumes each document has a "choices" field
    containing the list of answer choices and that the answer label symbols
    are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(f":[\s]*({without_paren_fallback_regex})")

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(fallback_regex, filter_ignores(resp), choice_to_alpha)
                    if not match:
                        match = find_match(without_paren_fallback_regex, resp, without_paren_to_target)
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


class ExtendedRegexFilter(RegexFilter):
    punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def filter_ignores(self, st):
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)

        if self.ignore_case:
            st = st.lower()

        if self.ignore_punctuation:
            # https://stackoverflow.com/a/266162
            st = st.translate(self.punct_tbl)
        return st

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match


# Designed for the AI2D/RealworldQA dataset
class SimpleMultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
