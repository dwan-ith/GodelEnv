"""Verifier-preserving evolution of GodelEnv's challenge curriculum.

The model may select source tasks and a bounded mutation operator, but it never
authors hidden references or reward code. This keeps environment evolution
separate from the solver and makes every accepted mutation replayable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from godel_engine.deterministic_solver import solve_task
from godel_engine.evolution import ADVANCED_STRATEGY_TEXT
from godel_engine.models import EnvironmentPatch, EnvironmentPatchDecision
from godel_engine.tasks.base import TaskInstance


ALLOWED_TASK_TYPES: frozenset[str] = frozenset({"factual_qa", "alignment_qa"})
ALLOWED_OPERATORS: frozenset[str] = frozenset({"deepen", "contrast", "transfer"})
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
    """Legacy reference for old free-text challenge API compatibility only."""
    words = re.findall(r"\b[a-zA-Z]{5,}\b", (prompt or "").lower())[:12]
    concept_groups: list[tuple[str, ...]] = [(word,) for word in words]
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


def validate_agent_challenge_proposal(task_type: str, prompt: str) -> tuple[bool, str | None]:
    """Validate the deprecated free-text challenge surface."""
    if task_type != "factual_qa":
        return False, "legacy free-text challenges are limited to factual_qa"
    text = (prompt or "").strip()
    if len(text) < _MIN_PROMPT_LEN:
        return False, f"prompt too short (min {_MIN_PROMPT_LEN} chars)"
    if len(text) > _MAX_PROMPT_LEN:
        return False, f"prompt too long (max {_MAX_PROMPT_LEN} chars)"
    lowered = text.lower()
    for bad in _FORBIDDEN_SUBSTR:
        if bad in lowered:
            return False, f"forbidden pattern in prompt: {bad!r}"
    return True, None


@dataclass
class EnvironmentGovernorConfig:
    min_novelty: float = 0.2
    min_solvability: float = 0.62
    min_current_score: float = 0.05
    max_current_score: float = 0.92
    min_regret: float = 0.005
    min_learning_value: float = 0.28


@dataclass
class PooledAgentChallenge:
    id: str
    task_type: str
    prompt: str
    source_episode: str = ""
    operator: str = "legacy"
    source_task_ids: list[str] = field(default_factory=list)
    reference: dict[str, Any] | None = None
    parent_id: str | None = None
    generation: int = 0
    novelty: float = 0.0
    solvability: float = 0.0
    current_strategy_score: float = 0.0
    regret: float = 0.0
    frontier_score: float = 0.0
    learning_value: float = 0.0
    fingerprint: str = ""
    evaluations: int = 0
    mean_solver_score: float = 0.0

    @property
    def priority(self) -> float:
        failure_signal = 1.0 - self.mean_solver_score if self.evaluations else 0.5
        return max(0.01, 0.45 * self.learning_value + 0.35 * failure_signal + 0.2 * self.novelty)


def _entry(task: Any, task_id: str) -> dict[str, Any] | None:
    return next(
        (dict(item) for item in getattr(task, "dataset", []) if str(item.get("id")) == task_id),
        None,
    )


def _merge_groups(entries: list[dict[str, Any]], key: str) -> list[Any]:
    merged: list[Any] = []
    for entry in entries:
        for group in entry.get(key, []):
            normalized = tuple(group) if isinstance(group, (list, tuple)) else (str(group),)
            if normalized not in merged:
                merged.append(normalized)
    return merged


def _build_reference(task_type: str, entries: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "id": "evolved",
        "prompt": prompt,
        "initial_solution": "",
        "agent_generated": True,
        "source_task_ids": [str(item["id"]) for item in entries],
    }
    if task_type == "factual_qa":
        for key in ("concept_groups", "contrast_left", "contrast_right"):
            reference[key] = _merge_groups(entries, key)
        reference["minimum_words"] = min(90, max(int(item.get("minimum_words", 40)) for item in entries) + 15)
        reference["target_words"] = min(180, sum(int(item.get("target_words", 80)) for item in entries))
    elif task_type == "alignment_qa":
        for key in ("concept_groups", "risk_groups", "mitigation_groups"):
            reference[key] = _merge_groups(entries, key)
        reference["minimum_words"] = min(100, max(int(item.get("minimum_words", 55)) for item in entries) + 15)
        reference["target_words"] = min(200, sum(int(item.get("target_words", 100)) for item in entries))
    else:
        raise ValueError(f"unsupported evolved task type: {task_type}")
    return reference


def _fingerprint(patch: EnvironmentPatch) -> str:
    payload = f"{patch.task_type}|{patch.operator}|{'|'.join(sorted(patch.source_task_ids))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class ChallengePool:
    """Versioned archive of accepted, verifier-backed environment mutations."""

    max_size: int = 32
    items: list[PooledAgentChallenge] = field(default_factory=list)
    last_rejection: str | None = None
    last_accepted_id: str | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    governor_config: EnvironmentGovernorConfig = field(default_factory=EnvironmentGovernorConfig)

    def try_add(self, *, task_type: str, prompt: str, source_episode: str = "") -> tuple[bool, str | None]:
        """Deprecated compatibility path; not counted as environment evolution evidence."""
        self.last_rejection = None
        self.last_accepted_id = None
        if os.getenv("GODEL_AGENT_CHALLENGES", "1").lower() in ("0", "false", "no"):
            return False, "agent challenges disabled (GODEL_AGENT_CHALLENGES=0)"
        ok, error = validate_agent_challenge_proposal(task_type, prompt)
        if not ok:
            self.last_rejection = error
            return False, error
        challenge_id = f"legacy_{uuid.uuid4().hex[:10]}"
        self.items.append(
            PooledAgentChallenge(
                id=challenge_id,
                task_type=task_type,
                prompt=prompt.strip(),
                source_episode=source_episode,
            )
        )
        self.last_accepted_id = challenge_id
        self._trim()
        return True, None

    async def evaluate_and_add(
        self,
        patch: EnvironmentPatch,
        *,
        tasks: dict[str, Any],
        strategy_evaluator: Any,
        strategy_text: str,
        source_episode: str = "",
    ) -> EnvironmentPatchDecision:
        """Apply independent validity, novelty, solvability, regret, and frontier gates."""
        self.proposed += 1
        reasons: list[str] = []
        self.last_rejection = None
        self.last_accepted_id = None

        if patch.task_type not in ALLOWED_TASK_TYPES:
            reasons.append(f"task_type must be one of {sorted(ALLOWED_TASK_TYPES)}")
        if patch.operator not in ALLOWED_OPERATORS:
            reasons.append(f"operator must be one of {sorted(ALLOWED_OPERATORS)}")
        expected_sources = 1 if patch.operator == "deepen" else 2
        if len(set(patch.source_task_ids)) != expected_sources:
            reasons.append(f"operator {patch.operator!r} requires {expected_sources} distinct source task(s)")

        task = tasks.get(patch.task_type)
        entries = [_entry(task, task_id) for task_id in patch.source_task_ids] if task else []
        if not task or any(entry is None for entry in entries):
            reasons.append("all source_task_ids must exist in the immutable task dataset")

        parent = None
        if patch.parent_challenge_id:
            parent = next((item for item in self.items if item.id == patch.parent_challenge_id), None)
            if parent is None:
                reasons.append("parent_challenge_id must reference an accepted challenge")

        fingerprint = _fingerprint(patch)
        if any(getattr(item, "fingerprint", None) == fingerprint for item in self.items):
            reasons.append("duplicate challenge genome")

        if reasons:
            return self._reject(reasons)

        typed_entries = [entry for entry in entries if entry is not None]
        source_prompts = [str(entry["prompt"]) for entry in typed_entries]
        if patch.operator == "deepen":
            prompt = (
                f"{source_prompts[0]} Go beyond a definition: explain the mechanism, "
                "identify one realistic failure mode or limitation, and finish with a verification check."
            )
        elif patch.operator == "contrast":
            prompt = (
                "Compare and connect these two verified problems. Explain each mechanism, the key "
                f"contrast, and one shared failure mode. (1) {source_prompts[0]} (2) {source_prompts[1]}"
            )
        else:
            prompt = (
                "Transfer the principle from the first verified problem to analyze the second. State "
                f"where the analogy works and where it breaks. (1) {source_prompts[0]} (2) {source_prompts[1]}"
            )

        reference = _build_reference(patch.task_type, typed_entries, prompt)
        challenge_id = f"env_{fingerprint}"
        candidate = PooledAgentChallenge(
            id=challenge_id,
            task_type=patch.task_type,
            prompt=prompt,
            source_episode=source_episode,
            operator=patch.operator,
            source_task_ids=list(patch.source_task_ids),
            reference=reference,
            parent_id=parent.id if parent else None,
            generation=(parent.generation + 1) if parent else 1,
        )
        candidate.fingerprint = fingerprint

        novelty = self._novelty(candidate)
        instance = self.materialize(candidate, tasks)
        current_solution, source = await strategy_evaluator._solve_case(
            task_prompt=instance.prompt,
            task_type=candidate.task_type,
            strategy_text=strategy_text,
            reference=reference,
        )
        current_score, _, _ = await task.grade(instance, current_solution)
        teacher_solution = solve_task(
            task_prompt=instance.prompt,
            task_type=candidate.task_type,
            strategy_text=ADVANCED_STRATEGY_TEXT,
            reference=reference,
        )
        teacher_score, _, _ = await task.grade(instance, teacher_solution)
        current_score = max(0.0, min(1.0, float(current_score)))
        solvability = max(0.0, min(1.0, float(teacher_score)))
        regret = max(0.0, solvability - current_score)
        target = float(patch.target_success_rate)
        frontier_score = max(0.0, 1.0 - abs(current_score - target) / max(target, 1.0 - target))
        learning_value = 0.3 * novelty + 0.4 * regret + 0.3 * frontier_score

        config = self.governor_config
        if novelty < config.min_novelty:
            reasons.append(f"novelty {novelty:.3f} below {config.min_novelty:.3f}")
        if solvability < config.min_solvability:
            reasons.append(f"solvability {solvability:.3f} below {config.min_solvability:.3f}")
        if current_score < config.min_current_score:
            reasons.append("challenge is currently too difficult to provide a learning signal")
        if current_score > config.max_current_score:
            reasons.append("challenge is already mastered")
        if regret < config.min_regret:
            reasons.append(f"teacher-current regret {regret:.3f} below {config.min_regret:.3f}")
        if learning_value < config.min_learning_value:
            reasons.append(f"learning value {learning_value:.3f} below {config.min_learning_value:.3f}")

        decision = EnvironmentPatchDecision(
            accepted=not reasons,
            challenge_id=challenge_id if not reasons else None,
            novelty=novelty,
            solvability=solvability,
            current_strategy_score=current_score,
            regret=regret,
            frontier_score=frontier_score,
            learning_value=learning_value,
            rejection_reasons=reasons,
            diagnostics={
                "solver_source": source,
                "operator": patch.operator,
                "source_task_ids": list(patch.source_task_ids),
                "verifier_owned_reference": True,
                "fingerprint": fingerprint,
            },
        )
        if decision.accepted:
            candidate.novelty = novelty
            candidate.solvability = solvability
            candidate.current_strategy_score = current_score
            candidate.regret = regret
            candidate.frontier_score = frontier_score
            candidate.learning_value = learning_value
            self.items.append(candidate)
            self.accepted += 1
            self.last_accepted_id = candidate.id
            self._trim()
        else:
            self.rejected += 1
            self.last_rejection = "; ".join(reasons)
        self.decisions.append(decision.model_dump(mode="json"))
        self.decisions = self.decisions[-50:]
        return decision

    def _reject(self, reasons: list[str]) -> EnvironmentPatchDecision:
        self.rejected += 1
        self.last_rejection = "; ".join(reasons)
        decision = EnvironmentPatchDecision(accepted=False, rejection_reasons=reasons)
        self.decisions.append(decision.model_dump(mode="json"))
        self.decisions = self.decisions[-50:]
        return decision

    def _novelty(self, candidate: PooledAgentChallenge) -> float:
        evolved = [item for item in self.items if item.operator != "legacy"]
        if not evolved:
            return 1.0
        candidate_tokens = _token_set(candidate.prompt)
        similarities: list[float] = []
        for item in evolved:
            tokens = _token_set(item.prompt)
            union = candidate_tokens | tokens
            similarities.append(len(candidate_tokens & tokens) / len(union) if union else 1.0)
        return max(0.0, 1.0 - max(similarities))

    def materialize(self, challenge: PooledAgentChallenge, tasks: dict[str, Any]) -> TaskInstance:
        task = tasks[challenge.task_type]
        reference = challenge.reference or synthetic_factual_reference(challenge.prompt)
        return TaskInstance(
            task_id=challenge.id,
            difficulty="evolved",
            prompt=challenge.prompt,
            initial_solution=str(reference.get("initial_solution", "")),
            reference=reference,
        )

    def record_evaluation(self, challenge_id: str, score: float) -> None:
        challenge = next((item for item in self.items if item.id == challenge_id), None)
        if challenge is None:
            return
        challenge.mean_solver_score = (
            challenge.mean_solver_score * challenge.evaluations + float(score)
        ) / (challenge.evaluations + 1)
        challenge.evaluations += 1

    def sample_for_eval(self, rng) -> PooledAgentChallenge | None:
        if not self.items:
            return None
        weights = [item.priority for item in self.items]
        return rng.choices(self.items, weights=weights, k=1)[0]

    def get(self, challenge_id: str) -> PooledAgentChallenge | None:
        return next((item for item in self.items if item.id == challenge_id), None)

    def _trim(self) -> None:
        while len(self.items) > self.max_size:
            self.items.remove(min(self.items, key=lambda item: item.priority))

    def as_stats(self) -> dict[str, Any]:
        evolved = [item for item in self.items if item.operator != "legacy"]
        return {
            "queued": len(self.items),
            "evolved_challenges": len(evolved),
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.accepted / self.proposed if self.proposed else 0.0,
            "max_generation": max((item.generation for item in evolved), default=0),
            "mean_learning_value": (
                sum(item.learning_value for item in evolved) / len(evolved) if evolved else 0.0
            ),
            "last_accepted_id": self.last_accepted_id,
            "last_rejection": self.last_rejection,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "max_size": self.max_size,
            "items": [asdict(item) for item in self.items],
            "decisions": list(self.decisions),
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "stats": self.as_stats(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChallengePool":
        pool = cls(max_size=int(payload.get("max_size", 32)))
        pool.items = [PooledAgentChallenge(**item) for item in payload.get("items", [])]
        pool.decisions = list(payload.get("decisions", []))
        pool.proposed = int(payload.get("proposed", 0))
        pool.accepted = int(payload.get("accepted", 0))
        pool.rejected = int(payload.get("rejected", 0))
        return pool

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChallengePool":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
