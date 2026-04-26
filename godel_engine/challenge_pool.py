"""
Agent-proposed challenges for curriculum expansion and self-play evaluation.

GodelEnv theme alignment:
- Agents can *propose* new study items; only validated prompts enter the pool.
- Held-out evaluation can mix fixed dataset cases with a sampled agent challenge
  (adaptive, growing task surface without abandoning verifiers).

Meta-strategy remains central: challenges stress-test the *current* strategy; they
do not replace StrategyPatch or Governor decisions.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field


# v1: text-only verifiable path via existing FactualQATask + synthetic reference.
ALLOWED_TASK_TYPES: frozenset[str] = frozenset({"factual_qa"})

_MIN_PROMPT_LEN = 20
_MAX_PROMPT_LEN = 2000
_FORBIDDEN_SUBSTR = (
    "rubric_scores",
    "total_score",
    "ignore previous",
    "system prompt",
    "you are now",
)


def synthetic_factual_reference(prompt: str) -> dict:
    """Build grading reference for agent-authored factual prompts (verifiable heuristics)."""
    words = re.findall(r"\b[a-zA-Z]{5,}\b", (prompt or "").lower())[:12]
    concept_groups: list[tuple[str, ...]] = [(w,) for w in words]
    if not concept_groups:
        concept_groups = [("explanation", "explain"), ("because",), ("conclusion",)]
    midpoint = max(1, len(concept_groups) // 2)
    n_words = max(len((prompt or "").split()), 1)
    return {
        "agent_generated": True,
        "id": "agent_synth",
        "prompt": prompt,
        "initial_solution": "",
        "concept_groups": concept_groups,
        "contrast_left": concept_groups[:midpoint],
        "contrast_right": concept_groups[midpoint:] or concept_groups[:midpoint],
        "minimum_words": min(40, max(20, n_words // 2)),
        "target_words": min(150, max(60, n_words * 2)),
    }


def validate_agent_challenge_proposal(
    task_type: str,
    prompt: str,
) -> tuple[bool, str | None]:
    if task_type not in ALLOWED_TASK_TYPES:
        return False, f"task_type must be one of {sorted(ALLOWED_TASK_TYPES)}"
    p = (prompt or "").strip()
    if len(p) < _MIN_PROMPT_LEN:
        return False, f"prompt too short (min {_MIN_PROMPT_LEN} chars)"
    if len(p) > _MAX_PROMPT_LEN:
        return False, f"prompt too long (max {_MAX_PROMPT_LEN} chars)"
    low = p.lower()
    for bad in _FORBIDDEN_SUBSTR:
        if bad in low:
            return False, f"forbidden pattern in prompt: {bad!r}"
    return True, None


@dataclass
class PooledAgentChallenge:
    id: str
    task_type: str
    prompt: str
    source_episode: str = ""


@dataclass
class ChallengePool:
    """In-memory store of agent proposals that passed validation (per env instance)."""

    max_size: int = 32
    items: list[PooledAgentChallenge] = field(default_factory=list)
    last_rejection: str | None = None
    last_accepted_id: str | None = None

    def try_add(
        self,
        *,
        task_type: str,
        prompt: str,
        source_episode: str = "",
    ) -> tuple[bool, str | None]:
        self.last_rejection = None
        self.last_accepted_id = None
        if os.getenv("GODEL_AGENT_CHALLENGES", "1").lower() in ("0", "false", "no"):
            return False, "agent challenges disabled (GODEL_AGENT_CHALLENGES=0)"
        ok, err = validate_agent_challenge_proposal(task_type, prompt)
        if not ok:
            self.last_rejection = err
            return False, err
        cid = f"agent_{uuid.uuid4().hex[:10]}"
        self.items.append(
            PooledAgentChallenge(
                id=cid,
                task_type=task_type,
                prompt=prompt.strip(),
                source_episode=source_episode,
            )
        )
        self.last_accepted_id = cid
        while len(self.items) > self.max_size:
            self.items.pop(0)
        return True, None

    def sample_for_eval(self, rng) -> PooledAgentChallenge | None:
        if not self.items:
            return None
        return rng.choice(self.items)

    def as_stats(self) -> dict[str, int | str | None]:
        return {
            "queued": len(self.items),
            "last_accepted_id": self.last_accepted_id,
            "last_rejection": self.last_rejection,
        }
