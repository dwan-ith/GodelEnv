"""Standalone inference runner for GodelEnv.

This script prefers a live OpenEnv websocket server when `GODEL_URL` is
available, but it can also run the environment in-process. That keeps
submission validation simple: `python inference.py` produces parseable
`[START]`, `[STEP]`, and `[END]` lines even when no local server is running.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Sequence

from dotenv import load_dotenv

from godel_engine.agent import AutoAgent
from godel_engine.client import GodelEngineEnv
from godel_engine.environment import GodelEnvironment
from godel_engine.models import GodelAction, GodelStepResult


load_dotenv(override=False)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("godel_env").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

DEFAULT_REMOTE_URL = "http://localhost:7860"
DEFAULT_TASKS = ("factual_qa", "code_improvement", "strategy_optimization")
REPRESENTATIVE_TASKS = {
    "easy": "factual_qa",
    "medium": "code_improvement",
    "hard": "adr_writing",
    "godel": "strategy_optimization",
}
KNOWN_TASKS = tuple(GodelEnvironment.TASKS.keys())
MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("OPENAI_MODEL_NAME")
    or os.getenv("HF_MODEL_NAME")
    or os.getenv("OLLAMA_MODEL_NAME")
    or "gpt-4o-mini"
)


def format_action(action: GodelAction) -> str:
    """Render a compact action summary that stays easy to parse from logs."""
    edit_type = action.edit_type.value if hasattr(action.edit_type, "value") else str(action.edit_type)
    safe_note = action.strategy_note.replace("\n", " ").replace("\r", "")
    return f"{edit_type}('{safe_note[:30]}')"


def sanitize_error(exc: Exception | str) -> str:
    return str(exc).replace("\n", " ").replace("\r", " ").strip() or "unknown_error"


def resolve_tasks(explicit_tasks: Sequence[str] | None) -> list[str]:
    raw_tasks = list(explicit_tasks or [])
    if not raw_tasks:
        env_tasks = os.getenv("GODEL_TASKS") or os.getenv("TASK")
        if env_tasks:
            raw_tasks = [part.strip() for part in env_tasks.split(",") if part.strip()]
    if not raw_tasks:
        return list(DEFAULT_TASKS)

    resolved: list[str] = []
    for item in raw_tasks:
        key = item.strip().lower()
        resolved.append(REPRESENTATIVE_TASKS.get(key, key))

    unknown = [task for task in resolved if task not in KNOWN_TASKS]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {', '.join(sorted(unknown))}. "
            f"Choose from: {', '.join(KNOWN_TASKS)}"
        )
    return resolved


class LocalGodelSession:
    """Tiny adapter that mirrors the remote client interface for local runs."""

    def __init__(self, seed: int | None = None) -> None:
        self._env = GodelEnvironment(seed=seed)

    async def __aenter__(self) -> "LocalGodelSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def reset(self, task_type: str = "", difficulty: str = "", task_id: str = "") -> GodelStepResult:
        return await self._env.reset(
            task_type=task_type or None,
            difficulty=difficulty or None,
            task_id=task_id or None,
        )

    async def step(self, action: GodelAction) -> GodelStepResult:
        return await self._env.step(action)


async def open_session(mode: str, remote_url: str, seed: int | None) -> tuple[object, str]:
    if mode not in {"auto", "local", "remote"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode in {"auto", "remote"}:
        client = GodelEngineEnv(remote_url)
        try:
            await client.__aenter__()
            return client, "remote"
        except Exception as exc:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
            if mode == "remote":
                raise RuntimeError(f"Remote env unavailable at {remote_url}: {sanitize_error(exc)}") from exc

    local = LocalGodelSession(seed=seed)
    await local.__aenter__()
    return local, "local"


async def close_session(session: object) -> None:
    close = getattr(session, "__aexit__", None)
    if close is None:
        return
    await close(None, None, None)


async def run_task(session: object, agent: AutoAgent, task_type: str) -> bool:
    print(f"[START] task={task_type} env=GodelEnv model={MODEL_NAME}")

    rewards: list[float] = []
    final_step = 0
    final_score = 0.001
    success = False
    final_error: str | None = None

    try:
        result = await session.reset(task_type=task_type)
        obs = result.observation
        last_step_result = result

        while not (result.terminated or result.truncated):
            inferred_rubrics = {name: f"Optimize {name}" for name in obs.rubric_scores.scores.keys()}
            action = await agent.act(
                task_prompt=obs.task_prompt,
                current_solution=obs.current_solution,
                rubrics=inferred_rubrics,
                task_type=task_type,
                strategy_text=obs.current_strategy,
                recent_failures=obs.recent_failures,
                downstream_scores=obs.downstream_scores,
            )

            try:
                result = await session.step(action)
                last_step_result = result
                error_msg = "null"
            except Exception as exc:
                result = last_step_result
                error_msg = sanitize_error(exc)
                final_error = error_msg
                reward_value = -1.0
                rewards.append(reward_value)
                print(
                    f"[STEP] step={obs.step} action={format_action(action)} "
                    f"reward={reward_value:.2f} done=true error={error_msg}"
                )
                break

            obs = result.observation
            reward_value = float(result.reward)
            rewards.append(reward_value)
            done = str(result.terminated or result.truncated).lower()
            print(
                f"[STEP] step={obs.step} action={format_action(action)} "
                f"reward={reward_value:.2f} done={done} error={error_msg}"
            )

        final_obs = result.observation
        final_step = int(final_obs.step)
        final_score = max(0.001, min(0.999, float(final_obs.total_score)))
        success = final_score >= 0.90
    except Exception as exc:
        final_error = sanitize_error(exc)

    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    error_suffix = f" error={final_error}" if final_error else ""
    print(
        f"[END] success={str(success).lower()} steps={final_step} "
        f"score={final_score:.3f} rewards={rewards_str}{error_suffix}"
    )
    return final_error is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GodelEnv baseline inference harness.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task names to run. Supports easy/medium/hard/godel shorthand.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "local", "remote"),
        default=os.getenv("GODEL_INFERENCE_MODE", "auto"),
        help="Prefer remote websocket env, local in-process env, or auto fallback.",
    )
    parser.add_argument(
        "--remote-url",
        default=os.getenv("GODEL_URL", DEFAULT_REMOTE_URL),
        help="Base URL for the websocket-backed OpenEnv server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for local runs.",
    )
    return parser.parse_args()


async def run_inference() -> int:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    agent = AutoAgent()
    session, _ = await open_session(args.mode, args.remote_url, args.seed)

    try:
        overall_ok = True
        for task_type in tasks:
            task_ok = await run_task(session, agent, task_type)
            overall_ok = overall_ok and task_ok
        return 0 if overall_ok else 1
    finally:
        await close_session(session)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_inference()))
