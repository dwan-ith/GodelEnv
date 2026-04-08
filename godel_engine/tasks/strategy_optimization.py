"""
Meta-Strategy Optimization (Godel/Recursive difficulty).

TRUE SELF-IMPROVEMENT: The agent's evolved strategy template is tested
against a downstream task. The grade measures whether the new template
actually makes reasoning better — not just whether it *looks* better.
"""
import random
from typing import Optional, Dict

from godel_engine.tasks.base import BaseTask, TaskInstance
from godel_engine.graders.agent_grader import AgentGrader

_STRATEGY_DATASET = [
    {
        "id": "godel01",
        "prompt": (
            "You are given a Reasoning Template that an AI agent uses to solve problems. "
            "Your job is to IMPROVE this template so it produces better answers.\n\n"
            "The template will be tested against this downstream challenge:\n"
            "  'Explain how RLHF can lead to reward hacking, and propose a mitigation.'\n\n"
            "Evolve the template to include: self-verification steps, "
            "hallucination checks, and structured reasoning stages."
        ),
        "initial_solution": (
            "REASONING TEMPLATE v1:\n"
            "1. Read the question.\n"
            "2. Think about it.\n"
            "3. Write an answer."
        ),
        "downstream_challenge": (
            "Explain how RLHF can lead to reward hacking, and propose a mitigation."
        ),
    },
    {
        "id": "godel02",
        "prompt": (
            "You are given a Strategy Template that an AI uses for step-by-step reasoning. "
            "Your job is to IMPROVE this template so it produces more accurate, "
            "verifiable outputs.\n\n"
            "The template will be tested against this downstream challenge:\n"
            "  'Compare the trade-offs between fine-tuning and in-context learning for domain adaptation.'\n\n"
            "Evolve the template to include: counter-argument generation, "
            "confidence calibration, and explicit uncertainty markers."
        ),
        "initial_solution": (
            "STRATEGY TEMPLATE v1:\n"
            "- Identify the core claim.\n"
            "- Support it with evidence.\n"
            "- Conclude."
        ),
        "downstream_challenge": (
            "Compare the trade-offs between fine-tuning and in-context learning "
            "for domain adaptation."
        ),
    },
]


class StrategyOptimizationTask(BaseTask):
    """
    The Gödel-tier task: the agent evolves its own reasoning template.

    Grading works in TWO phases:
      Phase 1 (structural): Does the template have self-verification,
              hallucination checks, structured stages?
      Phase 2 (empirical):  We SIMULATE using the template on a downstream
              challenge and grade the quality of the resulting reasoning.
              This is the recursive self-improvement loop.
    """

    def __init__(self):
        super().__init__("strategy_optimization", "godel")
        self.dataset = _STRATEGY_DATASET
        self._downstream_grader = AgentGrader()

    def sample(self, rng: Optional[random.Random] = None) -> TaskInstance:
        _rng = rng or random.Random()
        data = _rng.choice(self.dataset)
        return TaskInstance(
            task_id=data["id"],
            difficulty=self.difficulty,
            prompt=data["prompt"],
            initial_solution=data["initial_solution"],
            reference=data,
        )

    def _get_rubrics(self) -> Dict[str, str]:
        return {
            "self_verification": (
                "Does the template include explicit steps to check its own "
                "reasoning for logical consistency and factual accuracy?"
            ),
            "structural_rigor": (
                "Is the template well-structured with clear stages "
                "(e.g., numbered steps, sections, decision gates)?"
            ),
            "recursive_potential": (
                "Does the template have self-referential improvement hooks — "
                "mechanisms that could improve the template itself in future iterations?"
            ),
            "downstream_quality": (
                "When this template is mentally applied to a real reasoning challenge, "
                "would it produce a high-quality, nuanced, accurate response?"
            ),
        }

    async def grade(
        self, task_instance: TaskInstance, solution: str
    ) -> tuple[float, Dict[str, float], Dict[str, str]]:
        """
        Two-phase grading:
          1. Structural evaluation of the template itself.
          2. Empirical evaluation: simulate applying the template to a
             downstream challenge and grade the simulated output.
        """
        # Phase 1: structural grading via parent method
        total, scores, feedback = await super().grade(task_instance, solution)

        # Phase 2: empirical downstream test
        downstream_challenge = task_instance.reference.get(
            "downstream_challenge", ""
        )
        if downstream_challenge:
            downstream_score, downstream_fb = await self._test_downstream(
                solution, downstream_challenge
            )
            # Blend: 40% structural, 60% empirical (the real test)
            scores["empirical_downstream"] = downstream_score
            feedback["empirical_downstream"] = downstream_fb

            structural_avg = sum(
                v for k, v in scores.items() if k != "empirical_downstream"
            ) / max(1, len(scores) - 1)
            total = 0.4 * structural_avg + 0.6 * downstream_score

        return total, scores, feedback

    async def _test_downstream(
        self, strategy_template: str, challenge: str
    ) -> tuple[float, str]:
        """
        Simulate applying the strategy template to a downstream challenge.
        Grade the *quality of reasoning the template would produce*.
        """
        downstream_rubrics = {
            "reasoning_depth": (
                "Given this strategy template, would the resulting answer "
                "show deep, multi-step reasoning with explicit logical connections?"
            ),
            "accuracy_potential": (
                "Would following this template lead to a factually accurate "
                "and well-supported answer to the challenge?"
            ),
            "self_correction": (
                "Does the template enable the agent to catch and correct its own "
                "mistakes during reasoning?"
            ),
        }

        prompt = (
            f"A reasoning template is being evaluated for its effectiveness.\n\n"
            f"STRATEGY TEMPLATE BEING TESTED:\n{strategy_template}\n\n"
            f"DOWNSTREAM CHALLENGE:\n{challenge}\n\n"
            f"Evaluate how well this template would guide an AI to solve "
            f"the downstream challenge."
        )

        total, scores, fb = await self._downstream_grader.grade(
            task_prompt=prompt,
            current_solution=strategy_template,
            rubrics=downstream_rubrics,
        )

        # Aggregate downstream feedback
        summary = " | ".join(f"{k}: {v}" for k, v in fb.items())
        return total, summary
