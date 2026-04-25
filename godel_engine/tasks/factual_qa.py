"""
Deterministic factual explanation task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    joined_feedback,
    keyword_groups_score,
    length_score,
    missing_keyword_groups,
    paragraph_score,
    sentence_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


_QA_DATASET = [
    {
        "id": "qa01",
        "prompt": "What are the core differences between reinforcement learning and supervised learning?",
        "initial_solution": "RL is about agents. Supervised learning uses labeled data.",
        "concept_groups": [
            ("agent", "agents"),
            ("environment", "environments"),
            ("reward", "rewards"),
            ("action", "actions"),
            ("label", "labels", "labeled"),
            ("dataset", "data"),
            ("feedback", "supervision", "supervised signal"),
        ],
        "minimum_words": 45,
        "target_words": 85,
    },
    {
        "id": "qa02",
        "prompt": "Explain the significance of the attention mechanism in Transformers.",
        "initial_solution": "Attention lets the model look at useful parts of the sequence.",
        "concept_groups": [
            ("query", "queries"),
            ("key", "keys"),
            ("value", "values"),
            ("context", "contextual"),
            ("sequence", "token interactions"),
            ("parallel", "parallelism"),
        ],
        "minimum_words": 40,
        "target_words": 80,
    },
    {
        "id": "qa03",
        "prompt": "Explain quantum entanglement in simple terms.",
        "initial_solution": "Two particles stay connected across long distances.",
        "concept_groups": [
            ("particle", "particles"),
            ("state", "states"),
            ("measurement", "measuring"),
            ("correlated", "linked"),
            ("distance", "far apart"),
            ("quantum",),
        ],
        "minimum_words": 35,
        "target_words": 70,
    },
]


class FactualQATask(BaseTask):
    def __init__(self) -> None:
        super().__init__("factual_qa", "easy")
        self.dataset = _QA_DATASET

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
            "coverage": "Does the answer cover the core concepts needed for the explanation?",
            "detail": "Is the explanation concrete and complete rather than just a slogan?",
            "structure": "Is the explanation easy to follow with multiple coherent sentences?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        concept_groups = reference["concept_groups"]

        coverage = keyword_groups_score(solution, concept_groups)
        detail = (
            length_score(
                solution,
                minimum_words=reference["minimum_words"],
                target_words=reference["target_words"],
            )
            + sentence_score(solution, minimum_sentences=3)
        ) / 2
        structure = (
            sentence_score(solution, minimum_sentences=3)
            + paragraph_score(solution, minimum_paragraphs=1)
        ) / 2

        scores = {
            "coverage": coverage,
            "detail": detail,
            "structure": structure,
        }
        total = sum(scores.values()) / len(scores)

        feedback = {
            "coverage": joined_feedback(
                "Cover the central concepts more explicitly.",
                missing_keyword_groups(solution, concept_groups),
            ),
            "detail": (
                "Add more explanatory detail and concrete language."
                if detail < 0.95
                else "Detail level is strong."
            ),
            "structure": (
                "Use a clearer multi-sentence explanation."
                if structure < 0.95
                else "Structure is easy to follow."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
