from __future__ import annotations

import re
from typing import Iterable, Sequence


WORD_RE = re.compile(r"[a-zA-Z0-9_']+")
SENTENCE_RE = re.compile(r"[.!?]+")


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


def keyword_groups_score(text: str, groups: Sequence[str | Sequence[str]]) -> float:
    if not groups:
        return 1.0

    normalized = normalize_text(text)
    matched = 0
    for group in groups:
        aliases = _group_aliases(group)
        if any(alias.lower() in normalized for alias in aliases):
            matched += 1
    return matched / len(groups)


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


def joined_feedback(prefix: str, missing: Iterable[str]) -> str:
    missing_items = [item for item in missing if item]
    if not missing_items:
        return prefix
    return f"{prefix} Missing: {', '.join(missing_items)}."
