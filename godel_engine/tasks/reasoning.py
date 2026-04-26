"""
Deterministic structured reasoning task with broader scenario coverage.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    anti_boilerplate_score,
    balanced_pair_score,
    bullet_score,
    keyword_groups_score,
    semantic_specificity_score,
    section_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


def _make_item(
    *,
    item_id: str,
    prompt: str,
    initial_solution: str,
    pros_groups: list[tuple[str, ...] | tuple[str]],
    cons_groups: list[tuple[str, ...] | tuple[str]],
    recommendation_groups: list[tuple[str, ...] | tuple[str]],
) -> dict:
    return {
        "id": item_id,
        "prompt": prompt,
        "initial_solution": initial_solution,
        "pros_groups": pros_groups,
        "cons_groups": cons_groups,
        "recommendation_groups": recommendation_groups,
    }


_REASONING_DATASET = [
    _make_item(
        item_id="reason01",
        prompt="Evaluate the pros and cons of migrating from a monolith to microservices and provide a recommendation.",
        initial_solution="Microservices are better because they are isolated. They are harder to manage. I recommend doing it.",
        pros_groups=[
            ("scale", "scaling"),
            ("independent deployment", "deploy independently"),
            ("team autonomy", "independent teams"),
            ("fault isolation", "isolation"),
        ],
        cons_groups=[
            ("complexity", "operational complexity"),
            ("latency", "network latency"),
            ("distributed debugging", "debugging"),
            ("observability", "coordination overhead"),
        ],
        recommendation_groups=[
            ("recommend", "recommendation"),
            ("depends", "only if"),
            ("incremental", "phased", "gradual"),
        ],
    ),
    _make_item(
        item_id="reason02",
        prompt="Analyze the potential impact of artificial general intelligence (AGI) on global economics.",
        initial_solution="AGI could increase growth but also displace jobs. Governments should respond.",
        pros_groups=[
            ("productivity", "efficiency"),
            ("growth", "economic growth"),
            ("automation", "automate"),
            ("innovation", "new industries"),
        ],
        cons_groups=[
            ("job displacement", "displacement"),
            ("inequality", "concentrated power"),
            ("transition", "economic disruption"),
            ("policy risk", "regulatory risk"),
        ],
        recommendation_groups=[
            ("policy", "public policy"),
            ("adaptation", "reskilling"),
            ("regulation", "governance"),
        ],
    ),
    _make_item(
        item_id="reason03",
        prompt="Compare self-hosting a database with using a managed cloud database service.",
        initial_solution="Managed databases are easier but self-hosting gives more control.",
        pros_groups=[
            ("control", "customization"),
            ("cost savings", "lower cost"),
            ("compliance", "data residency"),
            ("predictable performance", "performance tuning"),
        ],
        cons_groups=[
            ("operations", "maintenance"),
            ("backups", "disaster recovery"),
            ("staffing", "expertise"),
            ("on-call", "incident response"),
        ],
        recommendation_groups=[
            ("recommend", "depends"),
            ("team size", "expertise"),
            ("regulatory", "compliance"),
        ],
    ),
    _make_item(
        item_id="reason04",
        prompt="Evaluate the trade-offs between using open-source foundation models and proprietary hosted models.",
        initial_solution="Open-source gives control while proprietary models are convenient.",
        pros_groups=[
            ("control", "customization"),
            ("privacy", "data control"),
            ("fine-tuning", "adaptation"),
            ("cost at scale", "unit economics"),
        ],
        cons_groups=[
            ("operations", "infrastructure"),
            ("latency risk", "serving complexity"),
            ("quality gap", "capability gap"),
            ("maintenance", "upgrades"),
        ],
        recommendation_groups=[
            ("hybrid", "mixed approach"),
            ("use case", "requirements"),
            ("compliance", "data sensitivity"),
        ],
    ),
    _make_item(
        item_id="reason05",
        prompt="Should a startup adopt event-driven architecture early or wait until its product and team mature?",
        initial_solution="Event-driven architecture scales, but it may be too much overhead early on.",
        pros_groups=[
            ("decoupling", "loose coupling"),
            ("scaling", "scale independently"),
            ("async", "asynchronous"),
            ("extensibility", "new consumers"),
        ],
        cons_groups=[
            ("complexity", "operational complexity"),
            ("debugging", "traceability"),
            ("ordering", "consistency"),
            ("premature", "too early"),
        ],
        recommendation_groups=[
            ("wait", "later", "when"),
            ("incremental", "phased"),
            ("depends", "based on"),
        ],
    ),
    _make_item(
        item_id="reason06",
        prompt="Compare in-office work with remote-first work for a distributed engineering organization.",
        initial_solution="Remote work gives flexibility but in-office work can help collaboration.",
        pros_groups=[
            ("flexibility",),
            ("hiring", "talent pool"),
            ("focus time", "deep work"),
            ("cost", "office cost"),
        ],
        cons_groups=[
            ("coordination", "communication"),
            ("mentorship", "onboarding"),
            ("culture", "team cohesion"),
            ("time zones", "timezone"),
        ],
        recommendation_groups=[
            ("hybrid",),
            ("intentional", "explicit process"),
            ("depends", "team needs"),
        ],
    ),
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
            "coverage": "Does the response cover both upside and downside with real domain details?",
            "balance": "Does the response treat both sides with similar depth?",
            "conclusion": "Does the response end with a clear recommendation or conditional next step?",
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
        coverage = (
            semantic_specificity_score(
                solution,
                list(reference["pros_groups"]) + list(reference["cons_groups"]),
                minimum_unique_groups=4,
                minimum_grounded_sentences=2,
            )
            + anti_boilerplate_score(
                solution,
                domain_groups=list(reference["pros_groups"]) + list(reference["cons_groups"]),
            )
        ) / 2
        balance = (balanced_pair_score(pros, cons) + (pros + cons) / 2) / 2
        conclusion = recommendation

        scores = {
            "structure": structure,
            "coverage": coverage,
            "balance": balance,
            "conclusion": conclusion,
        }
        total = (
            0.25 * structure
            + 0.3 * coverage
            + 0.2 * balance
            + 0.25 * conclusion
        )

        feedback = {
            "structure": (
                "Separate the response into pros, cons, and a recommendation."
                if structure < 0.95
                else "The reasoning structure is clear."
            ),
            "coverage": (
                "Use more task-specific reasoning instead of generic trade-off language."
                if coverage < 0.95
                else "The trade-offs are grounded in the domain."
            ),
            "balance": (
                "Cover both sides with similar depth."
                if balance < 0.95
                else "The trade-offs feel balanced."
            ),
            "conclusion": (
                "End with a concrete recommendation or condition."
                if conclusion < 0.95
                else "The conclusion is actionable."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
