"""
Deterministic Architecture Decision Record task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    keyword_groups_score,
    length_score,
    missing_keyword_groups,
    section_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


_ADR_DATASET = [
    {
        "id": "adr01",
        "prompt": "Evolve this sparse architecture note into a professional ADR for migrating a monolith to a serverless event-driven architecture.",
        "initial_solution": "Decided to move from monolith to AWS Lambda. Monolith is big and slow. Lambda is cheap and scales. Use SQS for events.",
        "required_sections": [
            ("title", "# title", "adr"),
            ("status",),
            ("context",),
            ("decision",),
            ("consequences",),
            ("alternatives",),
        ],
        "coverage_groups": [
            ("monolith", "existing system"),
            ("serverless", "aws lambda"),
            ("event-driven", "events"),
            ("queue", "sqs"),
            ("cost", "cost profile"),
            ("operations", "operational"),
            ("developer experience", "developer productivity", "team"),
        ],
        "tradeoff_groups": [
            ("benefit", "advantage", "positive"),
            ("risk", "drawback", "negative"),
            ("latency", "cold start"),
            ("observability", "debugging"),
        ],
    }
]


class ADRWritingTask(BaseTask):
    def __init__(self) -> None:
        super().__init__("adr_writing", "hard")
        self.dataset = _ADR_DATASET

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
            "structure": "Does the document follow a recognizable ADR structure?",
            "tradeoff_analysis": "Does the ADR explain both advantages and risks?",
            "consequences": "Does it discuss operational, cost, and developer-experience effects?",
            "clarity": "Is the document specific enough to guide an engineering decision?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        structure = section_score(solution, reference["required_sections"])
        tradeoff_analysis = keyword_groups_score(solution, reference["tradeoff_groups"])
        consequences = keyword_groups_score(
            solution,
            [
                ("operations", "operational", "on-call"),
                ("cost", "cost profile", "pricing"),
                ("developer experience", "developer productivity", "team autonomy"),
            ],
        )
        clarity = (
            keyword_groups_score(solution, reference["coverage_groups"])
            + length_score(solution, minimum_words=120, target_words=220)
        ) / 2

        scores = {
            "structure": structure,
            "tradeoff_analysis": tradeoff_analysis,
            "consequences": consequences,
            "clarity": clarity,
        }
        total = sum(scores.values()) / len(scores)

        feedback = {
            "structure": (
                "Use explicit ADR sections such as Status, Context, Decision, Consequences, and Alternatives."
                if structure < 0.95
                else "The ADR structure is solid."
            ),
            "tradeoff_analysis": (
                "Spell out both positive and negative trade-offs."
                if tradeoff_analysis < 0.95
                else "The trade-offs are well covered."
            ),
            "consequences": (
                "Expand the operational, cost, and developer-experience consequences."
                if consequences < 0.95
                else "The consequences section is concrete."
            ),
            "clarity": (
                "Add more specific architectural reasoning tied to the migration."
                if clarity < 0.95
                else "The decision rationale is specific and readable."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
