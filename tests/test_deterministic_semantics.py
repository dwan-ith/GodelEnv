from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["OPENAI_API_KEY"] = ""
os.environ["API_KEY"] = ""
os.environ["HF_TOKEN"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godel_engine.heuristic_policy import (
    build_heuristic_solution,
    build_heuristic_strategy_patch,
)
from godel_engine.tasks.alignment_qa import AlignmentQATask
from godel_engine.tasks.factual_qa import FactualQATask
from godel_engine.tasks.reasoning import ReasoningTask
from godel_engine.tasks.strategy_optimization import StrategyOptimizationTask


def test_text_task_datasets_are_not_tiny_anymore() -> None:
    assert len(FactualQATask().dataset) >= 8
    assert len(AlignmentQATask().dataset) >= 6
    assert len(ReasoningTask().dataset) >= 6
    assert len(StrategyOptimizationTask().dataset) >= 6


def test_heuristic_solution_is_task_specific() -> None:
    rl_answer = build_heuristic_solution(
        "What are the core differences between reinforcement learning and supervised learning?",
        "factual_qa",
        strategy_text="1. Decompose. 2. Gather evidence. 3. Verify.",
    ).lower()
    attention_answer = build_heuristic_solution(
        "Explain the significance of the attention mechanism in Transformers.",
        "factual_qa",
        strategy_text="1. Decompose. 2. Gather evidence. 3. Verify.",
    ).lower()

    assert "reward" in rl_answer or "environment" in rl_answer
    assert "query" in attention_answer or "key" in attention_answer
    assert "strategy hash" not in rl_answer
    assert "applying reasoning strategy" not in attention_answer


def test_factual_grader_penalizes_generic_boilerplate() -> None:
    async def _run() -> None:
        task = FactualQATask()
        instance = task.sample(task_id="qa01")
        generic = (
            "Analysis: Breaking down the key components. "
            "Evidence: Grounding the response in established knowledge and reasoning. "
            "Verification: Checking that the solution addresses the question."
        )
        grounded = (
            "Reinforcement learning trains an agent that takes actions in an environment and learns from rewards. "
            "Supervised learning instead maps inputs to labels from a dataset with direct supervision and immediate feedback. "
            "The key contrast is that RL learns from interaction and delayed reward, while supervised learning learns from labeled examples."
        )
        generic_score, _, _ = await task.grade(instance, generic)
        grounded_score, _, _ = await task.grade(instance, grounded)
        assert grounded_score > generic_score + 0.2

    asyncio.run(_run())


def test_strategy_task_requires_real_demonstration_substance() -> None:
    async def _run() -> None:
        task = StrategyOptimizationTask()
        instance = task.sample(task_id="godel01")
        generic = (
            "## Improved Strategy\n"
            "1. Analyze the problem.\n"
            "2. Use evidence.\n"
            "3. Verify the answer.\n"
            "4. Improve the strategy.\n\n"
            "## Demonstration\n"
            "This challenge is about important factors and systematic analysis. "
            "A good answer uses multiple perspectives and established reasoning."
        )
        grounded = (
            "## Improved Strategy\n"
            "1. Decompose the task into claims and assumptions.\n"
            "2. Gather evidence for each major claim.\n"
            "3. Generate a counterargument or edge case.\n"
            "4. Mark uncertainty where evidence is weaker.\n"
            "5. Verify the answer before finalizing.\n"
            "6. Revise the strategy when the same weakness repeats.\n\n"
            "## Demonstration\n"
            "RLHF can lead to reward hacking because the model optimizes a proxy reward rather than the full human objective. "
            "That makes it possible to exploit shortcuts that score well under the proxy reward while still being misaligned. "
            "A mitigation is to combine monitoring, auditing, and holdout evaluation so the system is checked by multiple independent signals."
        )
        generic_score, _, _ = await task.grade(instance, generic)
        grounded_score, _, _ = await task.grade(instance, grounded)
        assert grounded_score > generic_score + 0.2

    asyncio.run(_run())


def test_deterministic_patch_targets_real_weaknesses() -> None:
    patch = build_heuristic_strategy_patch(
        strategy_text="1. Read the task. 2. Answer directly.",
        recent_failures=["reasoning remained shallow and missed trade-offs"],
        downstream_scores={"reasoning": 0.32, "alignment_qa": 0.61},
    )
    improved = patch.improved_strategy.lower()
    targets = " ".join(patch.target_weaknesses).lower()
    assert "decompose" in improved or "counterargument" in improved
    assert "reasoning" in targets or "trade-off" in improved or "trade-off" in targets
