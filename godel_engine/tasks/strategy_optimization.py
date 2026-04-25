"""
Deterministic meta-strategy optimization task.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    bullet_score,
    keyword_groups_score,
    length_score,
    section_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


_STRATEGY_DATASET = [
    {
        "id": "godel01",
        "prompt": (
            "You are given a reasoning template that an AI agent uses to solve problems. "
            "Improve the template so it produces better answers.\n\n"
            "Your output must contain two sections:\n"
            "1. `## Improved Strategy` with the revised template.\n"
            "2. `## Demonstration` showing the strategy applied to this downstream challenge:\n"
            "   Explain how RLHF can lead to reward hacking, and propose a mitigation.\n"
        ),
        "initial_solution": (
            "REASONING TEMPLATE v1:\n"
            "1. Read the question.\n"
            "2. Think about it.\n"
            "3. Write an answer."
        ),
        "required_strategy_groups": [
            ("assumption", "assumptions"),
            ("verify", "verification", "check"),
            ("counterargument", "counter-example", "alternative hypothesis"),
            ("uncertainty", "confidence"),
            ("final answer", "answer"),
        ],
        "recursive_groups": [
            ("revise the strategy", "update the strategy", "improve the template"),
            ("lessons learned", "postmortem", "reflection"),
        ],
        "downstream_groups": [
            ("reward hacking", "gaming"),
            ("proxy reward", "proxy objective", "misspecified reward"),
            ("exploit", "shortcut"),
            ("mitigation", "monitoring", "auditing", "holdout"),
        ],
    },
    {
        "id": "godel02",
        "prompt": (
            "You are given a reasoning template that an AI agent uses for step-by-step analysis. "
            "Improve it so it produces more accurate, verifiable outputs.\n\n"
            "Your output must contain two sections:\n"
            "1. `## Improved Strategy` with the revised template.\n"
            "2. `## Demonstration` showing the strategy applied to this downstream challenge:\n"
            "   Compare the trade-offs between fine-tuning and in-context learning for domain adaptation.\n"
        ),
        "initial_solution": (
            "STRATEGY TEMPLATE v1:\n"
            "- Identify the claim.\n"
            "- Support it.\n"
            "- Conclude."
        ),
        "required_strategy_groups": [
            ("decompose", "break the problem down"),
            ("evidence", "supporting evidence"),
            ("counterargument", "alternative"),
            ("uncertainty", "confidence"),
            ("self-check", "verify", "verification"),
        ],
        "recursive_groups": [
            ("revise the strategy", "improve the template"),
            ("feedback loop", "iteration", "next revision"),
        ],
        "downstream_groups": [
            ("fine-tuning",),
            ("in-context learning", "icl"),
            ("domain adaptation",),
            ("data", "examples", "labels"),
            ("cost", "latency", "maintenance"),
            ("trade-off", "tradeoff"),
        ],
    },
]


class StrategyOptimizationTask(BaseTask):
    def __init__(self) -> None:
        super().__init__("strategy_optimization", "godel")
        self.dataset = _STRATEGY_DATASET

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
            "self_verification": "Does the strategy explicitly check its own reasoning?",
            "structural_rigor": "Is the strategy organized into clear, reusable stages?",
            "recursive_potential": "Does it explain how the strategy itself can improve over time?",
            "downstream_quality": "Does the demonstration solve the downstream challenge with substance?",
            "empirical_downstream": "Does the response actually apply the improved strategy rather than only describing it?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        normalized = solution.lower()
        has_strategy_section = "## improved strategy" in normalized
        has_demo_section = "## demonstration" in normalized

        self_verification = keyword_groups_score(solution, reference["required_strategy_groups"])
        structural_rigor = (
            section_score(
                solution,
                [
                    ("## improved strategy",),
                    ("## demonstration",),
                ],
            )
            + bullet_score(solution, minimum_bullets=4)
        ) / 2
        recursive_potential = keyword_groups_score(solution, reference["recursive_groups"])
        downstream_quality = keyword_groups_score(solution, reference["downstream_groups"])
        empirical_downstream = (
            0.5 * float(has_strategy_section and has_demo_section)
            + 0.5
            * length_score(solution, minimum_words=120, target_words=220)
        )

        scores = {
            "self_verification": self_verification,
            "structural_rigor": structural_rigor,
            "recursive_potential": recursive_potential,
            "downstream_quality": downstream_quality,
            "empirical_downstream": empirical_downstream,
        }
        total = (
            0.2 * self_verification
            + 0.2 * structural_rigor
            + 0.2 * recursive_potential
            + 0.2 * downstream_quality
            + 0.2 * empirical_downstream
        )

        feedback = {
            "self_verification": (
                "Add explicit self-check and uncertainty steps."
                if self_verification < 0.95
                else "The strategy includes explicit self-verification."
            ),
            "structural_rigor": (
                "Use the required sections and clearer numbered or bulleted stages."
                if structural_rigor < 0.95
                else "The strategy is well structured."
            ),
            "recursive_potential": (
                "Explain how the strategy learns from mistakes and revises itself."
                if recursive_potential < 0.95
                else "The strategy contains a real self-improvement loop."
            ),
            "downstream_quality": (
                "Strengthen the worked example on the downstream challenge."
                if downstream_quality < 0.95
                else "The demonstration covers the downstream task well."
            ),
            "empirical_downstream": (
                "Include both the improved strategy and a worked demonstration."
                if empirical_downstream < 0.95
                else "The strategy is empirically demonstrated."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
