from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["GODEL_AGENT_CHALLENGES"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godel_engine.challenge_pool import (
    ChallengePool,
    validate_agent_challenge_proposal,
)
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
