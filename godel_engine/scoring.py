from __future__ import annotations

import re
from typing import Iterable, Sequence


WORD_RE = re.compile(r"[a-zA-Z0-9_']+")
SENTENCE_RE = re.compile(r"[.!?]+")
GENERIC_FILLER_PATTERNS = (
    "key components",
    "important factors",
    "various aspects",
    "systematic analysis",
    "this task requires",
    "applying reasoning strategy",
    "established knowledge and reasoning",
    "multiple perspectives",
    "the primary analysis holds",
    "addresses the actual question",
)
CONTRAST_CONNECTORS = (
    "but",
    "while",
    "whereas",
    "instead",
    "however",
    "in contrast",
    "on the other hand",
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    count = len([chunk for chunk in SENTENCE_RE.split(text) if chunk.strip()])
    return max(1, count)


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def bullet_count(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*]|\d+\.)\s+", text))


def _group_aliases(group: str | Sequence[str]) -> list[str]:
    if isinstance(group, str):
        return [group]
    return [item for item in group if item]


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [chunk.strip() for chunk in raw if chunk.strip()]


def content_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if len(token) >= 4]


def matched_group_indices(text: str, groups: Sequence[str | Sequence[str]]) -> set[int]:
    normalized = normalize_text(text)
    matched: set[int] = set()
    for index, group in enumerate(groups):
        aliases = _group_aliases(group)
        if any(alias.lower() in normalized for alias in aliases):
            matched.add(index)
    return matched


def keyword_groups_score(text: str, groups: Sequence[str | Sequence[str]]) -> float:
    if not groups:
        return 1.0

    return len(matched_group_indices(text, groups)) / len(groups)


def missing_keyword_groups(
    text: str, groups: Sequence[str | Sequence[str]]
) -> list[str]:
    normalized = normalize_text(text)
    missing: list[str] = []
    for group in groups:
        aliases = _group_aliases(group)
        if not any(alias.lower() in normalized for alias in aliases):
            missing.append(aliases[0])
    return missing


def section_score(text: str, sections: Sequence[str | Sequence[str]]) -> float:
    return keyword_groups_score(text, sections)


def length_score(
    text: str,
    *,
    minimum_words: int,
    target_words: int | None = None,
    maximum_words: int | None = None,
) -> float:
    count = word_count(text)
    if count <= 0:
        return 0.0

    if count < minimum_words:
        return clamp(count / max(minimum_words, 1))

    if maximum_words is not None and count > maximum_words:
        overflow = count - maximum_words
        penalty_window = max(maximum_words // 2, 1)
        return clamp(1.0 - (overflow / penalty_window))

    if target_words is None:
        return 1.0

    if count >= target_words:
        return 1.0

    progress = (count - minimum_words) / max(target_words - minimum_words, 1)
    return clamp(0.7 + 0.3 * progress)


def sentence_score(text: str, minimum_sentences: int = 3) -> float:
    count = sentence_count(text)
    return clamp(count / max(minimum_sentences, 1))


def paragraph_score(text: str, minimum_paragraphs: int = 2) -> float:
    count = paragraph_count(text)
    return clamp(count / max(minimum_paragraphs, 1))


def bullet_score(text: str, minimum_bullets: int = 2) -> float:
    count = bullet_count(text)
    return clamp(count / max(minimum_bullets, 1))


def balanced_pair_score(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    spread = abs(left - right)
    return clamp(1.0 - spread)


def sentence_grounding_score(
    text: str,
    groups: Sequence[str | Sequence[str]],
    *,
    minimum_sentences: int = 2,
    minimum_group_hits_per_sentence: int = 1,
) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        hits = 0
        lowered = normalize_text(sentence)
        for group in groups:
            aliases = _group_aliases(group)
            if any(alias.lower() in lowered for alias in aliases):
                hits += 1
        if hits >= minimum_group_hits_per_sentence and word_count(sentence) >= 6:
            grounded += 1
    return clamp(grounded / max(minimum_sentences, 1))


def contrast_score(
    text: str,
    left_groups: Sequence[str | Sequence[str]],
    right_groups: Sequence[str | Sequence[str]],
    *,
    connectors: Sequence[str] = CONTRAST_CONNECTORS,
) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0

    for sentence in sentences:
        lowered = normalize_text(sentence)
        left = any(any(alias.lower() in lowered for alias in _group_aliases(group)) for group in left_groups)
        right = any(any(alias.lower() in lowered for alias in _group_aliases(group)) for group in right_groups)
        linked = any(connector in lowered for connector in connectors)
        if left and right and linked:
            return 1.0

    global_left = keyword_groups_score(text, left_groups)
    global_right = keyword_groups_score(text, right_groups)
    if global_left > 0.0 and global_right > 0.0:
        return 0.5
    return 0.0


def semantic_specificity_score(
    text: str,
    groups: Sequence[str | Sequence[str]],
    *,
    minimum_unique_groups: int = 4,
    minimum_grounded_sentences: int = 2,
) -> float:
    unique_matches = len(matched_group_indices(text, groups))
    unique_score = clamp(unique_matches / max(minimum_unique_groups, 1))
    grounded_score = sentence_grounding_score(
        text,
        groups,
        minimum_sentences=minimum_grounded_sentences,
    )
    return clamp(0.6 * unique_score + 0.4 * grounded_score)


def repetition_ratio(text: str) -> float:
    words = content_words(text)
    if len(words) < 10:
        return 0.0
    trigrams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    counts: dict[tuple[str, str, str], int] = {}
    for trigram in trigrams:
        counts[trigram] = counts.get(trigram, 0) + 1
    return max(counts.values()) / len(trigrams)


def anti_boilerplate_score(
    text: str,
    *,
    domain_groups: Sequence[str | Sequence[str]] = (),
) -> float:
    normalized = normalize_text(text)
    filler_hits = sum(1 for pattern in GENERIC_FILLER_PATTERNS if pattern in normalized)
    filler_penalty = min(1.0, filler_hits * 0.18)
    repetition_penalty = min(1.0, repetition_ratio(text) * 2.0)

    group_bonus = 0.0
    if domain_groups:
        group_bonus = 0.5 * semantic_specificity_score(
            text,
            domain_groups,
            minimum_unique_groups=max(2, min(4, len(domain_groups))),
            minimum_grounded_sentences=2,
        )
    return clamp(1.0 - filler_penalty - repetition_penalty + group_bonus, 0.0, 1.0)


def joined_feedback(prefix: str, missing: Iterable[str]) -> str:
    missing_items = [item for item in missing if item]
    if not missing_items:
        return prefix
    return f"{prefix} Missing: {', '.join(missing_items)}."
