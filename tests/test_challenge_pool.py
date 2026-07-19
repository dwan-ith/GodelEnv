from __future__ import annotations

import os
import random
import asyncio
import sys
from pathlib import Path

os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["GODEL_STRATEGY_EVAL_MODE"] = "deterministic"
os.environ["GODEL_ALLOW_DETERMINISTIC_FALLBACK"] = "1"
os.environ["GODEL_AGENT_CHALLENGES"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godel_engine.challenge_pool import (
    ChallengePool,
    validate_agent_challenge_proposal,
)
from godel_engine.environment import GodelEnvironment
from godel_engine.models import EnvironmentPatch
from godel_engine.strategy_evaluator import StrategyEvaluator
from godel_engine.tasks.factual_qa import FactualQATask


def test_validation_rejects_short_prompt() -> None:
    ok, err = validate_agent_challenge_proposal("factual_qa", "x" * 10)
    assert not ok
    assert err


def test_validation_rejects_bad_task_type() -> None:
    ok, err = validate_agent_challenge_proposal(
        "code_improvement",
        "x" * 30 + " enough padding for min length on purpose here.",
    )
    assert not ok
    assert err


def test_pool_try_add_and_sample() -> None:
    pool = ChallengePool()
    p = "Explain the photoelectric effect with experimental evidence. " * 2
    ok, e = pool.try_add(task_type="factual_qa", prompt=p, source_episode="e1")
    assert ok and e is None
    assert len(pool.items) == 1
    rng = random.Random(0)
    s = pool.sample_for_eval(rng)
    assert s is not None
    assert s.task_type == "factual_qa"
    stats = pool.as_stats()
    assert stats["queued"] == 1


def test_build_bundle_mixes_agent_factual_case() -> None:
    """Held-out slot can be replaced with a pooled agent-authored factual case."""
    tasks = {"factual_qa": FactualQATask()}
    ev = StrategyEvaluator(seed=0, max_cases=8)
    pool = ChallengePool()
    p = "Explain the photoelectric effect with experimental evidence. " * 2
    assert pool.try_add(task_type="factual_qa", prompt=p, source_episode="e1")[0]
    bundle = ev.build_bundle(tasks, episode_id="epmix001", challenge_pool=pool)
    agent_gens = [c for c in bundle if c.split == "agent_gen" and c.inline_prompt]
    assert len(agent_gens) >= 1, [c for c in bundle]


def test_agent_challenges_env_disable() -> None:
    os.environ["GODEL_AGENT_CHALLENGES"] = "0"
    try:
        pool = ChallengePool()
        p = "Explain the photoelectric effect with experimental evidence. " * 2
        ok, err = pool.try_add(task_type="factual_qa", prompt=p, source_episode="e1")
        assert not ok
        assert err and "disabled" in (err or "").lower()
    finally:
        os.environ["GODEL_AGENT_CHALLENGES"] = "1"


def test_environment_patch_is_governed_by_regret_and_persists(tmp_path) -> None:
    async def _run() -> tuple[ChallengePool, str]:
        env = GodelEnvironment(seed=42)
        await env.reset(
            task_type="strategy_optimization",
            task_id="godel02",
            seed=42,
            episode_id="environment-evolution-test",
        )
        patch = EnvironmentPatch(
            task_type="alignment_qa",
            operator="deepen",
            source_task_ids=["align02"],
            rationale="Expose reward misspecification blind spots.",
        )
        decision = await env.challenge_pool.evaluate_and_add(
            patch,
            tasks=env.tasks,
            strategy_evaluator=env.strategy_evaluator,
            strategy_text=env.current_strategy.policy_text,
            source_episode="test",
        )
        assert decision.accepted
        assert decision.solvability >= decision.current_strategy_score
        assert decision.regret > 0
        assert decision.diagnostics["verifier_owned_reference"] is True
        return env.challenge_pool, decision.challenge_id or ""

    pool, challenge_id = asyncio.run(_run())
    archive = tmp_path / "challenge_archive.json"
    pool.save(archive)
    restored = ChallengePool.load(archive)
    assert restored.get(challenge_id) is not None
    assert restored.as_stats()["max_generation"] == 1


def test_environment_patch_rejects_invalid_and_duplicate_genomes() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=42)
        await env.reset(task_type="strategy_optimization", task_id="godel02", seed=42)
        patch = EnvironmentPatch(
            task_type="alignment_qa",
            operator="deepen",
            source_task_ids=["align02"],
        )
        first = await env.challenge_pool.evaluate_and_add(
            patch,
            tasks=env.tasks,
            strategy_evaluator=env.strategy_evaluator,
            strategy_text=env.current_strategy.policy_text,
        )
        assert first.accepted
        duplicate = await env.challenge_pool.evaluate_and_add(
            patch,
            tasks=env.tasks,
            strategy_evaluator=env.strategy_evaluator,
            strategy_text=env.current_strategy.policy_text,
        )
        assert not duplicate.accepted
        assert any("duplicate" in reason for reason in duplicate.rejection_reasons)

        invalid = EnvironmentPatch(
            task_type="factual_qa",
            operator="deepen",
            source_task_ids=["missing"],
        )
        rejected = await env.challenge_pool.evaluate_and_add(
            invalid,
            tasks=env.tasks,
            strategy_evaluator=env.strategy_evaluator,
            strategy_text=env.current_strategy.policy_text,
        )
        assert not rejected.accepted
        assert any("immutable task dataset" in reason for reason in rejected.rejection_reasons)

    asyncio.run(_run())
