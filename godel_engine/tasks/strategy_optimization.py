"""
Deterministic meta-strategy optimization task with semantic downstream grading.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from godel_engine.scoring import (
    anti_boilerplate_score,
    bullet_score,
    keyword_groups_score,
    semantic_specificity_score,
    sentence_grounding_score,
    section_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


def _make_item(
    *,
    item_id: str,
    downstream_prompt: str,
    downstream_groups: list[tuple[str, ...] | tuple[str]],
) -> dict:
    prompt = (
        "You are given a reasoning template that an AI agent uses to solve problems. "
        "Improve the template so it produces better answers.\n\n"
        "Your output must contain two sections:\n"
        "1. `## Improved Strategy` with the revised template.\n"
        "2. `## Demonstration` showing the strategy applied to this downstream challenge:\n"
        f"   {downstream_prompt}\n"
    )
    return {
        "id": item_id,
        "prompt": prompt,
        "initial_solution": (
            "REASONING TEMPLATE v1:\n"
            "1. Read the question.\n"
            "2. Think about it.\n"
            "3. Write an answer."
        ),
        "required_strategy_groups": [
            ("decompose", "break the problem down", "claims"),
            ("evidence", "supporting evidence", "ground"),
            ("counterargument", "alternative"),
            ("uncertainty", "confidence"),
            ("verify", "self-check", "verification"),
        ],
        "recursive_groups": [
            ("revise the strategy", "update the strategy", "improve the template"),
            ("reflection", "lessons learned", "feedback loop", "iteration"),
        ],
        "downstream_groups": downstream_groups,
    }


_STRATEGY_DATASET = [
    _make_item(
        item_id="godel01",
        downstream_prompt="Explain how RLHF can lead to reward hacking, and propose a mitigation.",
        downstream_groups=[
            ("reward hacking", "gaming"),
            ("proxy reward", "proxy objective", "misspecified reward"),
            ("exploit", "shortcut"),
            ("mitigation", "monitoring", "auditing", "holdout"),
        ],
    ),
    _make_item(
        item_id="godel02",
        downstream_prompt="Compare the trade-offs between fine-tuning and in-context learning for domain adaptation.",
        downstream_groups=[
            ("fine-tuning",),
            ("in-context learning", "icl"),
            ("domain adaptation",),
            ("data", "examples", "labels"),
            ("cost", "latency", "maintenance"),
            ("trade-off", "tradeoff"),
        ],
    ),
    _make_item(
        item_id="godel03",
        downstream_prompt="Explain why retrieval-augmented generation can reduce hallucinations and where it still fails.",
        downstream_groups=[
            ("retrieval", "retrieve", "search"),
            ("hallucination", "fabricate", "made up"),
            ("grounding", "grounded"),
            ("context window", "documents", "knowledge base"),
            ("failure", "still fails", "limitation"),
        ],
    ),
    _make_item(
        item_id="godel04",
        downstream_prompt="Evaluate the trade-offs between a single large reward model and multiple independent verifiers.",
        downstream_groups=[
            ("reward model",),
            ("verifier", "verifiers"),
            ("gaming", "reward hacking"),
            ("independent checks", "multiple signals"),
            ("trade-off", "tradeoff"),
        ],
    ),
    _make_item(
        item_id="godel05",
        downstream_prompt="Explain why uncertainty estimates matter when an AI system gives medical triage advice.",
        downstream_groups=[
            ("uncertainty", "confidence"),
            ("medical", "triage"),
            ("risk", "harm"),
            ("defer", "escalate", "human review"),
            ("calibration", "trust"),
        ],
    ),
    _make_item(
        item_id="godel06",
        downstream_prompt="Compare training-time compute bottlenecks with inference-time compute bottlenecks for LLM systems.",
        downstream_groups=[
            ("training", "optimizer step"),
            ("inference", "rollout", "sampling"),
            ("compute", "latency"),
            ("batch size", "throughput"),
            ("bottleneck",),
        ],
    ),
]


def _extract_section(text: str, heading: str, next_heading: str | None = None) -> str:
    pattern = rf"{re.escape(heading)}\s*(.*)"
    if next_heading is not None:
        pattern = rf"{re.escape(heading)}\s*(.*?){re.escape(next_heading)}"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


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
        strategy_section = _extract_section(solution, "## Improved Strategy", "## Demonstration")
        demo_section = _extract_section(solution, "## Demonstration")

        self_verification = keyword_groups_score(strategy_section or solution, reference["required_strategy_groups"])
        structural_rigor = (
            section_score(
                solution,
                [
                    ("## improved strategy",),
                    ("## demonstration",),
                ],
            )
            + bullet_score(strategy_section or solution, minimum_bullets=4)
        ) / 2
        recursive_potential = keyword_groups_score(strategy_section or solution, reference["recursive_groups"])
        downstream_quality = (
            keyword_groups_score(demo_section or solution, reference["downstream_groups"])
            + semantic_specificity_score(
                demo_section or solution,
                reference["downstream_groups"],
                minimum_unique_groups=max(3, min(5, len(reference["downstream_groups"]))),
                minimum_grounded_sentences=2,
            )
            + anti_boilerplate_score(
                demo_section or solution,
                domain_groups=reference["downstream_groups"],
            )
        ) / 3
        empirical_downstream = (
            0.35 * float(has_strategy_section and has_demo_section)
            + 0.35
            * sentence_grounding_score(
                demo_section or solution,
                reference["downstream_groups"],
                minimum_sentences=2,
            )
            + 0.3
            * semantic_specificity_score(
                strategy_section or solution,
                reference["required_strategy_groups"],
                minimum_unique_groups=3,
                minimum_grounded_sentences=2,
            )
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
                "Strengthen the worked example with more task-specific substance."
                if downstream_quality < 0.95
                else "The demonstration covers the downstream challenge well."
            ),
            "empirical_downstream": (
                "Apply the improved strategy concretely instead of only describing it."
                if empirical_downstream < 0.95
                else "The strategy is empirically demonstrated."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
