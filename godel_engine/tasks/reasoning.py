"""
Deterministic structured reasoning task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    balanced_pair_score,
    bullet_score,
    keyword_groups_score,
    missing_keyword_groups,
    section_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


_REASONING_DATASET = [
    {
        "id": "reason01",
        "prompt": "Evaluate the pros and cons of migrating from a monolith to microservices and provide a recommendation.",
        "initial_solution": "Microservices are better because they are isolated. They are harder to manage. I recommend doing it.",
        "pros_groups": [
            ("scale", "scaling"),
            ("independent deployment", "deploy independently"),
            ("team autonomy", "independent teams"),
            ("fault isolation", "isolation"),
        ],
        "cons_groups": [
            ("complexity", "operational complexity"),
            ("latency", "network latency"),
            ("distributed debugging", "debugging"),
            ("observability", "coordination overhead"),
        ],
        "recommendation_groups": [
            ("recommend", "recommendation"),
            ("depends", "only if"),
            ("incremental", "phased", "gradual"),
        ],
    },
    {
        "id": "reason02",
        "prompt": "Analyze the potential impact of artificial general intelligence (AGI) on global economics.",
        "initial_solution": "AGI could increase growth but also displace jobs. Governments should respond.",
        "pros_groups": [
            ("productivity", "efficiency"),
            ("growth", "economic growth"),
            ("automation", "automate"),
            ("innovation", "new industries"),
        ],
        "cons_groups": [
            ("job displacement", "displacement"),
            ("inequality", "concentrated power"),
            ("transition", "economic disruption"),
            ("policy risk", "regulatory risk"),
        ],
        "recommendation_groups": [
            ("policy", "public policy"),
            ("adaptation", "reskilling"),
            ("regulation", "governance"),
        ],
    },
]


class ReasoningTask(BaseTask):
    def __init__(self) -> None:
        super().__init__("reasoning", "medium")
        self.dataset = _REASONING_DATASET

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
            "structure": "Does the response explicitly separate trade-offs and recommendation?",
            "balance": "Does the response cover both upside and downside with similar depth?",
            "conclusion": "Does the response end with a clear recommendation or next step?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        pros = keyword_groups_score(solution, reference["pros_groups"])
        cons = keyword_groups_score(solution, reference["cons_groups"])
        recommendation = keyword_groups_score(solution, reference["recommendation_groups"])

        structure = (
            section_score(
                solution,
                [
                    ("pros", "benefits", "upside"),
                    ("cons", "risks", "downsides"),
                    ("recommendation", "recommend", "conclusion"),
                ],
            )
            + bullet_score(solution, minimum_bullets=2)
        ) / 2
        balance = (balanced_pair_score(pros, cons) + (pros + cons) / 2) / 2
        conclusion = recommendation

        scores = {
            "structure": structure,
            "balance": balance,
            "conclusion": conclusion,
        }
        total = sum(scores.values()) / len(scores)

        feedback = {
            "structure": (
                "Separate the response into pros, cons, and a recommendation."
                if structure < 0.95
                else "The reasoning structure is clear."
            ),
            "balance": (
                "Cover both sides with similar depth."
                if balance < 0.95
                else "The trade-offs feel balanced."
            ),
            "conclusion": (
                "End with a concrete recommendation."
                if conclusion < 0.95
                else "The conclusion is actionable."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
