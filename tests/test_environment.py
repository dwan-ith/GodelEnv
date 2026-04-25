from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["HF_TOKEN"] = ""
os.environ["HF_API_KEY"] = ""
os.environ["HUGGINGFACE_API_KEY"] = ""
os.environ["HUGGINGFACE_TOKEN"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
os.environ["HF_ACCESS_TOKEN"] = ""
os.environ["API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from godel_engine.environment import GodelEnvironment
from godel_engine.evolution import Strategy
from godel_engine.heuristic_policy import build_heuristic_action
from godel_engine.provider_runtime import (
    DEFAULT_HF_ROUTER_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    ProviderCircuitBreaker,
    load_provider_configs,
)
from godel_engine.rollout import (
    build_prompt,
    collect_local_prompt_dataset,
    parse_completion_to_action,
    parse_prompt_metadata,
)
from godel_engine.symbolic_actions import ACTION_DIRECT_BEST, ACTION_PATCH_BALANCED
from server.app import app


def test_reset_can_target_specific_task_instance() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=123)
        result = await env.reset(task_type="factual_qa", task_id="qa02", seed=123)
        assert result.observation.task_type == "factual_qa"
        assert result.observation.task_id == "qa02"
        assert result.observation.difficulty == "easy"

    asyncio.run(_run())


def test_heuristic_policy_improves_selected_tasks() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=42)
        for task in ["factual_qa", "python_optimized", "adr_writing"]:
            start = await env.reset(task_type=task, seed=42)
            action = build_heuristic_action(start.observation.task_prompt, task)
            step = await env.step(action)
            assert step.observation.total_score > start.observation.total_score

    asyncio.run(_run())


def test_prompt_dataset_includes_replay_metadata() -> None:
    prompt_data = collect_local_prompt_dataset(num_prompts=2, tasks=["factual_qa"], seed=7)
    assert len(prompt_data) == 2
    meta = parse_prompt_metadata(prompt_data[0]["prompt"])
    assert meta["task_type"] == "factual_qa"
    assert meta["task_id"]
    assert "strategy_text" in prompt_data[0]


def test_openenv_websocket_session_preserves_state() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_type": "factual_qa", "task_id": "qa01"}})
        reset_payload = ws.receive_json()
        assert reset_payload["type"] == "observation"
        assert reset_payload["data"]["observation"]["step"] == 0

        ws.send_json(
            {
                "type": "step",
                "data": {
                    "solution": (
                        "Reinforcement learning trains an agent that acts in an environment, "
                        "collects rewards, and updates its policy from feedback over time. "
                        "Supervised learning instead learns from labeled examples in a dataset."
                    ),
                    "edit_type": "rewrite",
                    "strategy_note": "test step",
                },
            }
        )
        step_payload = ws.receive_json()
        assert step_payload["type"] == "observation"
        assert step_payload["data"]["observation"]["step"] == 1
        assert step_payload["data"]["observation"]["task_id"] == "qa01"


def test_openenv_websocket_accepts_strategy_patch() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "reset",
                "data": {"task_type": "strategy_optimization", "task_id": "godel01"},
            }
        )
        reset_payload = ws.receive_json()
        obs = reset_payload["data"]["observation"]

        action = build_heuristic_action(
            obs["task_prompt"],
            obs["task_type"],
            strategy_text=obs["current_strategy"],
        )
        assert action.strategy_patch is not None

        ws.send_json(
            {
                "type": "step",
                "data": action.model_dump(mode="json"),
            }
        )
        step_payload = ws.receive_json()
        assert step_payload["type"] == "observation"
        step_data = step_payload["data"]
        observation = step_data["observation"]
        assert observation["patch_decision"] is not None
        assert observation["patch_decision"]["tasks_evaluated"] > 0
        assert "patch_quality" in observation["reward_breakdown"]
        assert "generalization_score" in observation["reward_breakdown"]


def test_demo_act_endpoint_returns_action() -> None:
    client = TestClient(app)
    response = client.post(
        "/demo/act",
        json={
            "task_prompt": "Explain the difference between RL and supervised learning.",
            "current_solution": "RL is about agents.",
            "task_type": "factual_qa",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "solution" in payload
    assert payload["solution"]


def test_demo_provider_status_endpoint_returns_safe_provider_info() -> None:
    client = TestClient(app)
    response = client.get("/demo/provider-status")
    assert response.status_code == 200
    payload = response.json()
    assert "providers" in payload
    assert "grading_mode" in payload
    assert "strategy_eval_mode" in payload


def test_strategy_evaluation_depends_on_strategy_text() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=42)
        weak = Strategy(id="weak", policy_text="Read the question and answer it.")
        strong = Strategy(
            id="strong",
            policy_text=(
                "1. Decompose the problem into claims.\n"
                "2. Gather evidence and supporting details.\n"
                "3. Generate a counterargument or alternative.\n"
                "4. Mark uncertainty.\n"
                "5. Run a self-check and verify the answer.\n"
                "6. Reflect on repeated failures."
            ),
        )
        weak_axes, _, _ = await env._evaluate_strategy_downstream(weak)
        strong_axes, _, _ = await env._evaluate_strategy_downstream(strong)
        assert strong_axes["correctness"] > weak_axes["correctness"]
        assert strong_axes["generalization"] >= weak_axes["generalization"]

    asyncio.run(_run())


def test_compact_direct_action_token_expands_to_real_solution() -> None:
    action = parse_completion_to_action(
        ACTION_DIRECT_BEST,
        task_prompt="Explain the significance of the attention mechanism in Transformers.",
        task_type="factual_qa",
        current_solution="Attention helps.",
    )
    assert action.strategy_patch is None
    assert "attention" in action.solution.lower()


def test_compact_patch_action_token_expands_to_strategy_patch() -> None:
    action = parse_completion_to_action(
        ACTION_PATCH_BALANCED,
        task_prompt=(
            "Improve a reasoning template and demonstrate it on a downstream challenge."
        ),
        task_type="strategy_optimization",
        current_solution="Use the old template.",
        strategy_text="Read the question and answer it.",
    )
    assert action.strategy_patch is not None
    # The heuristic patch adds what's MISSING from the current strategy.
    # Since "Read the question and answer it" is missing most capabilities,
    # the patch should add at least one improvement (varies based on priority).
    improved = action.strategy_patch.improved_strategy.lower()
    has_improvement = any(
        keyword in improved
        for keyword in ["decompose", "verify", "evidence", "counter", "uncertainty", "reflect", "example", "efficiency", "safety"]
    )
    assert has_improvement, f"Expected an improvement keyword in: {improved}"


def test_compact_prompt_stays_within_training_budget() -> None:
    prompt = build_prompt(
        task_prompt="Explain the core differences between RL and supervised learning.",
        current_solution="RL is about agents.",
        rubric_feedback="Add rewards, environment interaction, and dataset labels.",
        task_type="strategy_optimization",
        task_id="godel02",
        strategy_text="Break the problem down, gather evidence, verify, and reflect.",
        downstream_scores={"factual_qa": 0.72, "reasoning": 0.61},
        recent_failures=["Missed uncertainty handling.", "Skipped the counterargument."],
    )
    assert "ACTION TOKEN:" in prompt
    assert len(prompt) < 2500


def test_provider_circuit_breaker_disables_after_connection_error() -> None:
    ProviderCircuitBreaker.reset()
    ProviderCircuitBreaker.record_failure("openai", RuntimeError("Connection error to provider"))
    assert ProviderCircuitBreaker.is_disabled("openai")
    assert not ProviderCircuitBreaker.is_disabled("huggingface")
    ProviderCircuitBreaker.reset()


def test_load_provider_configs_keeps_provider_groups_separate(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "custom-key")
    monkeypatch.setenv("API_BASE_URL", "https://custom.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)

    configs = load_provider_configs()
    assert [config.name for config in configs][:2] == ["custom", "openai"]
    openai_config = next(config for config in configs if config.name == "openai")
    assert openai_config.base_url is None


def test_load_provider_configs_defaults_hf_router(monkeypatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    configs = load_provider_configs()
    huggingface_config = next(config for config in configs if config.name == "huggingface")
    assert huggingface_config.base_url == DEFAULT_HF_ROUTER_BASE_URL


def test_load_provider_configs_uses_space_style_hf_base_url(monkeypatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    configs = load_provider_configs()
    huggingface_config = configs[0]
    assert huggingface_config.name == "huggingface"
    assert huggingface_config.base_url == "https://router.huggingface.co/v1"
    assert huggingface_config.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_load_provider_configs_accepts_hf_aliases(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_API_KEY", "hf-token")
    monkeypatch.setenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct:novita")

    configs = load_provider_configs()
    huggingface_config = configs[0]
    assert huggingface_config.name == "huggingface"
    assert huggingface_config.base_url == DEFAULT_HF_ROUTER_BASE_URL
    assert huggingface_config.model_name == "Qwen/Qwen2.5-7B-Instruct:novita"


def test_openai_never_uses_hub_model_id_from_model_name(monkeypatch) -> None:
    """MODEL_NAME is for HuggingFace; do not pass Qwen/... to the OpenAI API."""
    for key in (
        "HF_TOKEN",
        "HF_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_ACCESS_TOKEN",
        "API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)

    configs = load_provider_configs()
    openai_config = next(c for c in configs if c.name == "openai")
    assert openai_config.model_name == DEFAULT_OPENAI_MODEL
    assert "/" not in openai_config.model_name


# ---------------------------------------------------------------------------
# Governor and Guards Tests
# ---------------------------------------------------------------------------

from godel_engine.evolution import Governor, GovernorConfig
from godel_engine.guards import (
    GuardResult,
    canary_guard,
    run_strategy_guards,
    strategy_length_guard,
    strategy_regression_gate,
    strategy_variance_penalty,
)


def test_governor_accepts_patch_with_improvement() -> None:
    """Governor should accept patches that improve utility without major regressions."""
    governor = Governor()
    parent_scores = {"correctness": 0.6, "generalization": 0.5, "robustness": 0.5, "cost": 0.7, "stability": 0.6}
    child_scores = {"correctness": 0.7, "generalization": 0.55, "robustness": 0.55, "cost": 0.7, "stability": 0.65}
    per_task_parent = {"factual_qa": 0.6, "reasoning": 0.5, "alignment_qa": 0.5}
    per_task_child = {"factual_qa": 0.65, "reasoning": 0.55, "alignment_qa": 0.6}

    decision = governor.decide(parent_scores, child_scores, per_task_parent, per_task_child)
    assert decision["accepted"] is True
    assert decision["improvement"] > 0


def test_governor_rejects_patch_with_too_many_regressions() -> None:
    """Governor should reject patches that regress on too many tasks."""
    governor = Governor()
    parent_scores = {"correctness": 0.6, "generalization": 0.5, "robustness": 0.5, "cost": 0.7, "stability": 0.6}
    child_scores = {"correctness": 0.65, "generalization": 0.4, "robustness": 0.45, "cost": 0.7, "stability": 0.5}
    per_task_parent = {"factual_qa": 0.6, "reasoning": 0.5, "alignment_qa": 0.5, "code": 0.6}
    per_task_child = {"factual_qa": 0.7, "reasoning": 0.3, "alignment_qa": 0.3, "code": 0.4}

    decision = governor.decide(parent_scores, child_scores, per_task_parent, per_task_child)
    assert decision["accepted"] is False
    assert any("regression" in r.lower() for r in decision["rejection_reasons"])


def test_governor_rejects_patch_with_insufficient_improvement() -> None:
    """Governor should reject patches with negligible improvement."""
    config = GovernorConfig(min_improvement=0.01)
    governor = Governor(config)
    parent_scores = {"correctness": 0.6, "generalization": 0.5, "robustness": 0.5, "cost": 0.7, "stability": 0.6}
    child_scores = {"correctness": 0.601, "generalization": 0.5, "robustness": 0.5, "cost": 0.7, "stability": 0.6}
    per_task_parent = {"factual_qa": 0.6}
    per_task_child = {"factual_qa": 0.601}

    decision = governor.decide(parent_scores, child_scores, per_task_parent, per_task_child)
    assert decision["accepted"] is False
    assert any("improvement" in r.lower() for r in decision["rejection_reasons"])


def test_strategy_length_guard_rejects_short_strategies() -> None:
    """Strategies that are too short should be penalized."""
    short_strategy = "Be good at tasks."
    penalty, violation = strategy_length_guard(short_strategy)
    assert penalty < 0
    assert "too short" in violation.lower()


def test_strategy_length_guard_accepts_reasonable_strategies() -> None:
    """Strategies with reasonable length should pass."""
    reasonable_strategy = (
        "1. Decompose the problem into atomic claims and assumptions. "
        "2. Gather evidence for each claim from reliable sources. "
        "3. Generate counterarguments and alternative hypotheses. "
        "4. Mark confidence levels for uncertain claims. "
        "5. Run a self-check before finalizing the answer."
    )
    penalty, violation = strategy_length_guard(reasonable_strategy)
    assert penalty == 0.0
    assert violation is None


def test_canary_guard_catches_gaming_attempts() -> None:
    """Canary guard should detect strategies that reference internals."""
    gaming_strategy = (
        "To score well, maximize rubric_scores by including keywords "
        "that match the total_score calculation."
    )
    penalty, violation = canary_guard(gaming_strategy)
    assert penalty < 0
    assert "canary" in violation.lower()


def test_canary_guard_passes_legitimate_strategies() -> None:
    """Legitimate strategies should pass the canary guard."""
    good_strategy = (
        "1. Read and understand the task prompt carefully. "
        "2. Break down the problem into smaller components. "
        "3. Gather relevant evidence and verify claims. "
        "4. Generate alternative solutions and counterarguments. "
        "5. Self-check the final answer for accuracy."
    )
    penalty, violation = canary_guard(good_strategy)
    assert penalty == 0.0
    assert violation is None


def test_strategy_regression_gate_allows_some_regressions() -> None:
    """Regression gate should allow regressions up to the threshold."""
    per_task_parent = {"a": 0.6, "b": 0.6, "c": 0.6, "d": 0.6, "e": 0.6}
    per_task_child = {"a": 0.7, "b": 0.7, "c": 0.7, "d": 0.55, "e": 0.55}  # 2/5 = 40% regression
    
    # With 35% threshold, 40% should fail
    penalty, violation = strategy_regression_gate(per_task_parent, per_task_child, max_regression_fraction=0.35)
    assert penalty < 0
    assert "regressed" in violation.lower()
    
    # With 50% threshold, 40% should pass
    penalty, violation = strategy_regression_gate(per_task_parent, per_task_child, max_regression_fraction=0.50)
    assert penalty == 0.0
    assert violation is None


def test_run_strategy_guards_aggregates_penalties() -> None:
    """run_strategy_guards should aggregate all guard results."""
    # Very short strategy + high regression = multiple penalties
    short_strategy = "Do stuff."
    per_task_parent = {"a": 0.9, "b": 0.9, "c": 0.9}
    per_task_child = {"a": 0.1, "b": 0.1, "c": 0.1}

    result = run_strategy_guards(short_strategy, per_task_parent, per_task_child)
    assert result.passed is False
    assert len(result.violations) >= 2  # At least length + regression
    assert result.penalty < 0
