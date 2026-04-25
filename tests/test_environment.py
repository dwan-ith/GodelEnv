from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["HF_TOKEN"] = ""
os.environ["API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from godel_engine.environment import GodelEnvironment
from godel_engine.evolution import Strategy
from godel_engine.heuristic_policy import build_heuristic_action
from godel_engine.rollout import collect_local_prompt_dataset, parse_prompt_metadata
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
        weak_axes, _ = await env._evaluate_strategy_downstream(weak)
        strong_axes, _ = await env._evaluate_strategy_downstream(strong)
        assert strong_axes["correctness"] > weak_axes["correctness"]
        assert strong_axes["generalization"] >= weak_axes["generalization"]

    asyncio.run(_run())
