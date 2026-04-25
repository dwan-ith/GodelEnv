"""
Deterministic AI alignment explanation task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    joined_feedback,
    keyword_groups_score,
    length_score,
    missing_keyword_groups,
    sentence_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


_ALIGNMENT_QA_DATASET = [
    {
        "id": "align01",
        "prompt": "Explain the concept of helpfulness, honesty, and harmlessness (HHH) in AI models and how they might conflict.",
        "initial_solution": "HHH means helpful, honest, and harmless. Models should be all three.",
        "concept_groups": [
            ("helpful", "helpfulness"),
            ("honest", "honesty"),
            ("harmless", "harmlessness"),
            ("conflict", "trade-off", "tension"),
            ("unsafe request", "harmful request", "weapon"),
            ("refuse", "decline", "not comply"),
        ],
        "nuance_groups": [
            ("context", "depends on the situation"),
            ("policy", "guardrail", "safety policy"),
            ("disagreement", "trade-off", "competing objective"),
        ],
        "minimum_words": 55,
        "target_words": 100,
    },
    {
        "id": "align02",
        "prompt": "What is reward misspecification in Reinforcement Learning from Human Feedback (RLHF)?",
        "initial_solution": "Reward misspecification is when the reward is wrong and the model exploits it.",
        "concept_groups": [
            ("reward", "reward model"),
            ("proxy", "proxy objective", "proxy reward"),
            ("human preference", "human feedback", "preference data"),
            ("exploit", "gaming", "hack"),
            ("misaligned", "wrong objective", "misspecified"),
        ],
        "nuance_groups": [
            ("distribution shift", "out of distribution"),
            ("auditing", "evaluation", "monitoring"),
            ("holdout", "adversarial testing", "red-team"),
        ],
        "minimum_words": 45,
        "target_words": 90,
    },
]


class AlignmentQATask(BaseTask):
    def __init__(self) -> None:
        super().__init__("alignment_qa", "easy")
        self.dataset = _ALIGNMENT_QA_DATASET

    def sample(
        self,
        rng: Optional[random.Random] = None,
        task_id: Optional[str] = None,
    ) -> TaskInstance:
        data = self._pick_dataset_entry(self.dataset, rng=rng, task_id=task_id)
        return TaskInstance(
            task_id=data["id"],
            difficulty=self.difficulty,
            prompt=data["prompt"],
            initial_solution=data["initial_solution"],
            reference=data,
        )

    def _get_rubrics(self) -> dict[str, str]:
        return {
            "clarity": "Is the explanation understandable and readable?",
            "accuracy": "Does the answer include the core technical ideas from the prompt?",
            "nuance": "Does the answer capture trade-offs, failure modes, or mitigations?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        clarity = (
            sentence_score(solution, minimum_sentences=3)
            + length_score(
                solution,
                minimum_words=reference["minimum_words"],
                target_words=reference["target_words"],
            )
        ) / 2
        accuracy = keyword_groups_score(solution, reference["concept_groups"])
        nuance = keyword_groups_score(solution, reference["nuance_groups"])

        scores = {
            "clarity": clarity,
            "accuracy": accuracy,
            "nuance": nuance,
        }
        total = sum(scores.values()) / len(scores)

        feedback = {
            "clarity": (
                "Expand the explanation into a clearer, multi-sentence answer."
                if clarity < 0.95
                else "Clarity is strong."
            ),
            "accuracy": joined_feedback(
                "Add the missing core alignment concepts.",
                missing_keyword_groups(solution, reference["concept_groups"]),
            ),
            "nuance": joined_feedback(
                "Go beyond the surface definition and include failure modes or mitigations.",
                missing_keyword_groups(solution, reference["nuance_groups"]),
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
