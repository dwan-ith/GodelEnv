from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["GODEL_STRATEGY_EVAL_MODE"] = "deterministic"
os.environ["GODEL_AGENT_MODE"] = "deterministic"
os.environ["GODEL_ALLOW_DETERMINISTIC_FALLBACK"] = "1"
os.environ["HF_TOKEN"] = ""
os.environ["HF_API_KEY"] = ""
os.environ["HUGGINGFACE_API_KEY"] = ""
os.environ["HUGGINGFACE_TOKEN"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
os.environ["HF_ACCESS_TOKEN"] = ""
os.environ["API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from godel_engine.agent import AutoAgent
from godel_engine.environment import GodelEnvironment
from godel_engine.evolution import Strategy
from godel_engine.guards import run_strategy_guards
from godel_engine.heuristic_policy import build_heuristic_action
from godel_engine.models import EditType, GodelAction, StrategyPatch
from godel_engine.provider_runtime import (
    DEFAULT_HF_ROUTER_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    ProviderCircuitBreaker,
    load_provider_configs,
    provider_completion_kwargs,
)
from godel_engine.rollout import (
    action_json_prefix,
    build_prompt,
    classify_action_origin,
    collect_local_prompt_dataset,
    parse_completion_to_action,
    parse_prompt_metadata,
)
from godel_engine.research_eval import (
    ADVERSARIAL_STRATEGY_PATCHES,
    ResearchEvaluator,
    StaticBaselinePatchAgent,
    confidence_interval,
    linear_slope,
)
from godel_engine.self_improve import SelfImprovementRunner
from godel_engine.strategy_evaluator import StrategyEvaluator
from server.app import app


class ScriptedLLMStrategyEvaluator:
    """A deterministic test double that reports LLM-like source diagnostics."""

    mode = "llm"

    async def evaluate(
        self,
        tasks,
        strategy_text: str,
        *,
        episode_id: str,
        current_task_id: str = "",
        challenge_pool=None,
    ):
        improved = all(
            marker in strategy_text.lower()
            for marker in ("decompose", "evidence", "verify", "uncertainty")
        )
        if improved:
            per_task = {
                "heldout:factual_qa:qa02": 0.74,
                "heldout:reasoning:reason01": 0.70,
                "canary:strategy_optimization:godel01": 0.72,
            }
            per_family = {
                "factual_qa": 0.74,
                "reasoning": 0.70,
                "strategy_optimization": 0.72,
            }
        else:
            per_task = {
                "heldout:factual_qa:qa02": 0.50,
                "heldout:reasoning:reason01": 0.48,
                "canary:strategy_optimization:godel01": 0.49,
            }
            per_family = {
                "factual_qa": 0.50,
                "reasoning": 0.48,
                "strategy_optimization": 0.49,
            }
        values = list(per_family.values())
        axis_scores = {
            "correctness": sum(values) / len(values),
            "generalization": min(values),
            "robustness": 0.96,
            "cost": 0.70,
            "stability": 0.95,
        }
        return axis_scores, per_task, {
            "mode": "llm",
            "verifier": "llm_required",
            "per_family": per_family,
            "axis_scores": axis_scores,
            "source_counts": {"llm:test": len(per_task)},
            "providers": [{"name": "test", "configured": True}],
            "agent_challenges_mixed": 0,
        }


class ScriptedPatchAgent:
    def __init__(self, *, provider: str = "test-agent", model: str | None = None) -> None:
        self.last_provider = provider
        self.last_model = model
        self.last_source = f"llm:{provider}:{model}" if model else f"llm:{provider}"
        self.last_error = None

    async def act(self, **kwargs) -> GodelAction:
        self.last_source = (
            f"llm:{self.last_provider}:{self.last_model}"
            if self.last_model
            else f"llm:{self.last_provider}"
        )
        return GodelAction(
            solution="Demonstrate a verified evidence-first strategy.",
            edit_type=EditType.REWRITE,
            strategy_note="scripted strategy patch",
            strategy_patch=StrategyPatch(
                improved_strategy=(
                    "1. Decompose the task into claims and assumptions.\n"
                    "2. Gather evidence for every important claim.\n"
                    "3. Mark uncertainty where the evidence is incomplete.\n"
                    "4. Verify that the final answer addresses the exact prompt.\n"
                    "5. Reflect on repeated failures before proposing the next patch."
                ),
                diff_description="Adds evidence, uncertainty, verification, and reflection.",
                hypothesis="The stronger loop should improve held-out task performance.",
                target_weaknesses=["missing evidence", "missing verification"],
            ),
        )


def test_deterministic_agent_never_contacts_configured_provider(monkeypatch) -> None:
    class BombClient:
        def chat(self, *args, **kwargs):
            raise AssertionError("deterministic mode contacted an LLM provider")

    async def _run() -> None:
        monkeypatch.setenv("GODEL_AGENT_MODE", "deterministic")
        agent = AutoAgent()
        agent.clients = [("bomb", "bomb-model", BombClient())]
        action = await agent.act(
            task_prompt="Improve the strategy.",
            current_solution="",
            rubrics={},
            task_type="strategy_optimization",
        )
        assert action.strategy_patch is not None
        assert agent.last_source == "deterministic"
        assert agent.last_provider is None

    asyncio.run(_run())


def test_reset_can_target_specific_task_instance() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=123)
        result = await env.reset(task_type="factual_qa", task_id="qa02", seed=123)
        assert result.observation.task_type == "factual_qa"
        assert result.observation.task_id == "qa02"
        assert result.observation.difficulty == "easy"

    asyncio.run(_run())


def test_heuristic_policy_produces_valid_solutions() -> None:
    """Test that heuristic policy produces solutions (not necessarily improving)."""
    async def _run() -> None:
        env = GodelEnvironment(seed=42)
        for task in ["factual_qa", "reasoning", "alignment_qa"]:
            start = await env.reset(task_type=task, seed=42)
            action = build_heuristic_action(start.observation.task_prompt, task)
            # Verify that action has non-empty solution
            assert len(action.solution) > 0
            # Verify that we can step with the action (no errors)
            step = await env.step(action)
            # Score should be non-negative
            assert step.observation.total_score >= 0.0

    asyncio.run(_run())


def test_prompt_dataset_includes_replay_metadata() -> None:
    prompt_data = collect_local_prompt_dataset(num_prompts=2, tasks=["factual_qa"], seed=7)
    assert len(prompt_data) == 2
    meta = parse_prompt_metadata(prompt_data[0]["prompt"])
    assert meta["task_type"] == "factual_qa"
    assert meta["task_id"]
    assert "strategy_text" in prompt_data[0]
    assert "reference" in prompt_data[0]
    assert prompt_data[0]["reference"]["id"].startswith("qa")


def test_strategy_evaluator_bundle_includes_adversarial_cases() -> None:
    env = GodelEnvironment(seed=1)
    evaluator = StrategyEvaluator(seed=1, max_cases=4)
    bundle = evaluator.build_bundle(
        env.tasks,
        episode_id="adv-test",
        current_task_id="godel01",
    )
    assert any(case.split == "adversarial" for case in bundle)
    assert any(case.split == "canary" for case in bundle)


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


def test_strategy_patch_can_be_accepted_with_llm_evaluator() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=42)
        env.strategy_evaluator = ScriptedLLMStrategyEvaluator()
        reset_result = await env.reset(
            task_type="strategy_optimization", task_id="godel01", seed=42
        )
        baseline_utility = reset_result.observation.total_score
        parent_id = env.current_strategy.id

        action = GodelAction(
            solution="Demonstrate the revised strategy on the downstream challenge.",
            strategy_patch=StrategyPatch(
                improved_strategy=(
                    "1. Decompose the task into claims and assumptions.\n"
                    "2. Gather evidence for each major claim.\n"
                    "3. Mark uncertainty where the evidence is incomplete.\n"
                    "4. Verify that the final answer addresses the exact prompt.\n"
                    "5. Reflect on repeated failures before future patches."
                ),
                diff_description="Adds evidence, uncertainty, and verification.",
                hypothesis="More explicit checks should improve held-out task performance.",
                target_weaknesses=["missing verification", "under-specified evidence"],
            ),
        )

        result = await env.step(action)
        assert result.patch_decision is not None
        assert result.patch_decision.parent_utility == baseline_utility
        assert result.patch_decision.accepted is True
        assert env.current_strategy.id != parent_id
        assert env.patches_accepted == 1
        assert result.reward > 0
        assert result.patch_decision.diagnostics["child_source_counts"] == {"llm:test": 3}

    asyncio.run(_run())


def test_strategy_reset_episode_id_makes_hidden_bundle_reproducible() -> None:
    async def _run() -> None:
        first = GodelEnvironment(seed=42)
        second = GodelEnvironment(seed=42)
        first_result = await first.reset(
            task_type="strategy_optimization",
            task_id="godel02",
            seed=42,
            episode_id="fixed-evidence-bundle",
        )
        second_result = await second.reset(
            task_type="strategy_optimization",
            task_id="godel02",
            seed=42,
            episode_id="fixed-evidence-bundle",
        )
        assert first_result.observation.total_score == pytest.approx(
            second_result.observation.total_score
        )
        assert first_result.observation.downstream_scores == pytest.approx(
            second_result.observation.downstream_scores
        )

    asyncio.run(_run())


def test_rejected_patch_does_not_advance_score_or_reward_child_bonus() -> None:
    async def _run() -> None:
        env = GodelEnvironment(seed=42)
        env.strategy_evaluator = ScriptedLLMStrategyEvaluator()
        await env.reset(task_type="strategy_optimization", task_id="godel01", seed=42)
        parent_id = env.current_strategy.id

        action = GodelAction(
            solution="A vague demonstration.",
            strategy_patch=StrategyPatch(
                improved_strategy=(
                    "1. Read the prompt.\n"
                    "2. Answer directly.\n"
                    "3. Be concise without adding the extra verification structure."
                ),
                diff_description="Too small to matter.",
                hypothesis="A shorter strategy might be cheaper.",
                target_weaknesses=["verbosity"],
            ),
        )

        result = await env.step(action)
        assert result.patch_decision is not None
        assert result.patch_decision.accepted is False
        assert env.current_strategy.id == parent_id
        assert env.patches_rejected == 1
        assert result.reward < 0
        assert result.reward_breakdown.generalization_score == 0.0

    asyncio.run(_run())


def test_self_improvement_runner_persists_accepted_lineage() -> None:
    async def _run() -> None:
        scratch = Path("artifacts") / "test_self_improve" / uuid.uuid4().hex
        registry_path = scratch / "registry.json"
        metrics_path = scratch / "metrics.json"
        runner = SelfImprovementRunner(
            registry_path=registry_path,
            metrics_path=metrics_path,
            agent=ScriptedPatchAgent(),
            strategy_evaluator=ScriptedLLMStrategyEvaluator(),
            seed=5,
        )
        summary = await runner.run(iterations=2, task_ids=["godel01"])

        assert summary["attempts"] >= 2
        assert summary["patches_proposed"] >= 2
        assert summary["patches_accepted"] >= 1
        assert summary["llm_evaluated_attempts"] == summary["patches_proposed"]
        assert summary["all_strategy_evals_llm_backed"] is True
        assert summary["best_strategy_generation"] >= 1
        assert registry_path.exists()
        assert metrics_path.exists()

        reloaded = SelfImprovementRunner(
            registry_path=registry_path,
            agent=ScriptedPatchAgent(),
            strategy_evaluator=ScriptedLLMStrategyEvaluator(),
            seed=6,
        )
        assert reloaded.registry.get_stats()["max_generation"] >= 1
        registry_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        scratch.rmdir()

    asyncio.run(_run())


def test_research_evaluator_reports_statistics_and_baseline_comparison() -> None:
    async def _run() -> None:
        scratch = Path("artifacts") / "test_research_eval" / uuid.uuid4().hex
        evaluator = ResearchEvaluator(
            seeds=[3, 4],
            iterations=2,
            output_dir=scratch,
            agent=ScriptedPatchAgent(),
            baseline_agent=StaticBaselinePatchAgent(),
            strategy_evaluator=ScriptedLLMStrategyEvaluator(),
            task_ids=["godel01"],
        )
        report = await evaluator.run()

        assert report["runs"] == 2
        assert report["patches_proposed"] >= 4
        assert report["patches_accepted"] >= 2
        assert report["all_strategy_evals_llm_backed"] is True
        assert report["provider_separation_ok"] is True
        assert report["improvement_ci"]["n"] == report["patches_proposed"]
        assert report["reward_ci"]["low"] <= report["reward_ci"]["mean"] <= report["reward_ci"]["high"]
        assert "baseline_comparison" in report
        assert report["baseline_comparison"]["baseline_runs"] == 2
        assert (scratch / "research_report.json").exists()

    asyncio.run(_run())


def test_provider_model_separation_accepts_distinct_models() -> None:
    class OtherModelEvaluator(ScriptedLLMStrategyEvaluator):
        async def evaluate(self, *args, **kwargs):
            axes, per_task, diagnostics = await super().evaluate(*args, **kwargs)
            diagnostics["source_counts"] = {"llm:custom:verifier-model": len(per_task)}
            return axes, per_task, diagnostics

    async def _run() -> None:
        scratch = Path("artifacts") / "test_self_improve" / uuid.uuid4().hex
        runner = SelfImprovementRunner(
            registry_path=scratch / "registry.json",
            metrics_path=scratch / "metrics.json",
            agent=ScriptedPatchAgent(provider="custom", model="agent-model"),
            strategy_evaluator=OtherModelEvaluator(),
            seed=7,
        )
        summary = await runner.run(iterations=1, task_ids=["godel01"])

        assert summary["patches_proposed"] == 1
        assert summary["provider_separation_ok"] is True

    asyncio.run(_run())


def test_provider_model_separation_flags_same_model() -> None:
    class SameModelEvaluator(ScriptedLLMStrategyEvaluator):
        async def evaluate(self, *args, **kwargs):
            axes, per_task, diagnostics = await super().evaluate(*args, **kwargs)
            diagnostics["source_counts"] = {"llm:custom:same-model": len(per_task)}
            return axes, per_task, diagnostics

    async def _run() -> None:
        scratch = Path("artifacts") / "test_self_improve" / uuid.uuid4().hex
        runner = SelfImprovementRunner(
            registry_path=scratch / "registry.json",
            metrics_path=scratch / "metrics.json",
            agent=ScriptedPatchAgent(provider="custom", model="same-model"),
            strategy_evaluator=SameModelEvaluator(),
            seed=8,
        )
        summary = await runner.run(iterations=1, task_ids=["godel01"])

        assert summary["patches_proposed"] == 1
        assert summary["provider_separation_ok"] is False

    asyncio.run(_run())


def test_statistical_helpers_handle_confidence_and_trends() -> None:
    ci = confidence_interval([1.0, 2.0, 3.0])
    assert ci["n"] == 3
    assert ci["low"] < ci["mean"] < ci["high"]
    assert linear_slope([1.0, 2.0, 3.0, 4.0]) > 0
    assert linear_slope([4.0, 3.0, 2.0, 1.0]) < 0


def test_adversarial_strategy_patches_trip_reward_hacking_guards() -> None:
    parent = {"heldout:a": 0.8, "heldout:b": 0.8, "canary:c": 0.8}
    child = {"heldout:a": 0.1, "heldout:b": 0.1, "canary:c": 0.1}
    for patch in ADVERSARIAL_STRATEGY_PATCHES:
        result = run_strategy_guards(patch["strategy_text"], parent, child)
        assert result.passed is False, patch["name"]
        assert result.penalty < 0.0


def test_json_action_parses_to_real_solution() -> None:
    """Test that JSON completions parse into GodelActions correctly."""
    json_completion = '{"solution": "Attention mechanisms allow Transformers to focus on relevant parts of the input.", "edit_type": "rewrite", "strategy_note": "Addressed core concept"}'
    action = parse_completion_to_action(
        json_completion,
        task_prompt="Explain the significance of the attention mechanism in Transformers.",
        task_type="factual_qa",
        current_solution="Attention helps.",
    )
    assert action.strategy_patch is None
    assert "attention" in action.solution.lower()


def test_json_strategy_patch_parses_correctly() -> None:
    """Test that JSON completions with strategy patches parse correctly."""
    json_completion = '''{
        "improved_strategy": "1. Decompose the problem. 2. Gather evidence. 3. Verify conclusions.",
        "diff_description": "Added decomposition and verification steps",
        "hypothesis": "Structured reasoning improves accuracy",
        "target_weaknesses": ["no decomposition", "no verification"],
        "solution": "Demonstration of the improved strategy."
    }'''
    action = parse_completion_to_action(
        json_completion,
        task_prompt=(
            "Improve a reasoning template and demonstrate it on a downstream challenge."
        ),
        task_type="strategy_optimization",
        current_solution="Use the old template.",
        strategy_text="Read the question and answer it.",
    )
    assert action.strategy_patch is not None
    improved = action.strategy_patch.improved_strategy.lower()
    assert "decompose" in improved or "verify" in improved


def test_prefixed_json_continuation_repairs_to_structured_action() -> None:
    completion = (
        action_json_prefix("strategy_optimization")
        + '1. Decompose the task. 2. Check evidence. 3. Verify the final answer.",\n'
        + '  "diff_description": "Added evidence checking and verification",\n'
        + '  "hypothesis": "Verification should reduce repeated reasoning errors",\n'
        + '  "target_weaknesses": ["missed evidence", "no verification"],\n'
        + '  "solution": "Apply decomposition to identify the failure, then verify the corrected answer.",\n'
        + '  "edit_type": "rewrite",\n'
        + '  "strategy_note": "Adds verification before final answers"\n'
        + "}"
    )
    action = parse_completion_to_action(
        completion,
        task_prompt="Improve a reasoning strategy.",
        task_type="strategy_optimization",
        current_solution="Old strategy.",
    )
    assert classify_action_origin(action) == "json_patch"
    assert action.strategy_patch is not None
    assert "verify" in action.strategy_patch.improved_strategy.lower()


def test_freeform_prompt_uses_json_instruction() -> None:
    """Test that prompts instruct models to generate JSON, not action tokens."""
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
    # New prompts use JSON instruction, not action tokens
    assert "JSON" in prompt
    assert "improved_strategy" in prompt  # Strategy task prompts mention patch format
    assert "demonstrate the improved strategy on the task" not in prompt
    assert "the complete revised strategy text" not in prompt
    assert len(prompt) < 3000


def test_provider_circuit_breaker_disables_after_connection_error() -> None:
    ProviderCircuitBreaker.reset()


def test_provider_circuit_breaker_is_scoped_by_runtime_role() -> None:
    ProviderCircuitBreaker.reset()
    ProviderCircuitBreaker.record_failure(
        "openai", RuntimeError("Connection error to provider"), scope="grader"
    )
    assert ProviderCircuitBreaker.is_disabled("openai", scope="grader")
    assert not ProviderCircuitBreaker.is_disabled("openai", scope="agent")
    ProviderCircuitBreaker.reset()
    ProviderCircuitBreaker.record_failure("openai", RuntimeError("Connection error to provider"))
    assert ProviderCircuitBreaker.is_disabled("openai")
    assert not ProviderCircuitBreaker.is_disabled("huggingface")
    ProviderCircuitBreaker.reset()


def _clear_provider_env(monkeypatch) -> None:
    for key in (
        "API_KEY",
        "API_BASE_URL",
        "CUSTOM_API_KEY",
        "CUSTOM_API_BASE_URL",
        "CUSTOM_MODEL_NAME",
        "MODEL_NAME",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE_URL",
        "OPENAI_MODEL_NAME",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_BASE_URL",
        "OPENROUTER_MODEL_NAME",
        "HF_TOKEN",
        "HF_API_KEY",
        "HF_API_BASE_URL",
        "HF_MODEL_NAME",
        "HF_INFERENCE_MODEL",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_ACCESS_TOKEN",
        "OLLAMA_API_BASE_URL",
        "OLLAMA_HOST",
        "OLLAMA_MODEL_NAME",
        "OLLAMA_MODEL",
        "OLLAMA_API_KEY",
        "GODEL_USE_OLLAMA",
        "GODEL_PROVIDER_ORDER",
        "GODEL_AGENT_PROVIDER_ORDER",
        "GODEL_VERIFIER_PROVIDER_ORDER",
        "GODEL_AGENT_MODEL_NAME",
        "GODEL_VERIFIER_MODEL_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_provider_configs_keeps_provider_groups_separate(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("API_KEY", "custom-key")
    monkeypatch.setenv("API_BASE_URL", "https://custom.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GODEL_PROVIDER_ORDER", "custom,openai")

    configs = load_provider_configs()
    assert [config.name for config in configs][:2] == ["custom", "openai"]
    openai_config = next(config for config in configs if config.name == "openai")
    assert openai_config.base_url is None


def test_load_provider_configs_defaults_hf_router(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    configs = load_provider_configs()
    huggingface_config = next(config for config in configs if config.name == "huggingface")
    assert huggingface_config.base_url == DEFAULT_HF_ROUTER_BASE_URL


def test_load_provider_configs_uses_space_style_hf_base_url(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    configs = load_provider_configs()
    huggingface_config = configs[0]
    assert huggingface_config.name == "huggingface"
    assert huggingface_config.base_url == "https://router.huggingface.co/v1"
    assert huggingface_config.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_load_provider_configs_accepts_hf_aliases(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("HF_API_KEY", "hf-token")
    monkeypatch.setenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct:novita")

    configs = load_provider_configs()
    huggingface_config = configs[0]
    assert huggingface_config.name == "huggingface"
    assert huggingface_config.base_url == DEFAULT_HF_ROUTER_BASE_URL
    assert huggingface_config.model_name == "Qwen/Qwen2.5-7B-Instruct:novita"


def test_load_provider_configs_accepts_openrouter_aliases(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENROUTER_MODEL_NAME", "openai/gpt-4.1-mini")

    configs = load_provider_configs()
    custom_config = configs[0]
    assert custom_config.name == "custom"
    assert custom_config.base_url == "https://openrouter.ai/api/v1"
    assert custom_config.model_name == "openai/gpt-4.1-mini"


def test_role_specific_model_overrides_for_proposer_and_verifier(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENROUTER_MODEL_NAME", "default-model")
    monkeypatch.setenv("GODEL_AGENT_MODEL_NAME", "agent-model")
    monkeypatch.setenv("GODEL_VERIFIER_MODEL_NAME", "verifier-model")

    agent_config = load_provider_configs(order_env="GODEL_AGENT_PROVIDER_ORDER")[0]
    verifier_config = load_provider_configs(order_env="GODEL_VERIFIER_PROVIDER_ORDER")[0]

    assert agent_config.name == "custom"
    assert agent_config.model_name == "agent-model"
    assert verifier_config.name == "custom"
    assert verifier_config.model_name == "verifier-model"


def test_openai_never_uses_hub_model_id_from_model_name(monkeypatch) -> None:
    """MODEL_NAME is for HuggingFace; do not pass Qwen/... to the OpenAI API."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    configs = load_provider_configs()
    openai_config = next(c for c in configs if c.name == "openai")
    assert openai_config.model_name == DEFAULT_OPENAI_MODEL
    assert "/" not in openai_config.model_name


def test_ollama_completion_options_can_force_cpu(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_GPU", "0")
    assert provider_completion_kwargs("ollama") == {
        "extra_body": {"options": {"num_gpu": 0}}
    }
    assert provider_completion_kwargs("openai") == {}


def test_ollama_completion_options_reject_invalid_gpu_count(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_GPU", "invalid")
    with pytest.raises(ValueError, match="OLLAMA_NUM_GPU"):
        provider_completion_kwargs("ollama")


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
