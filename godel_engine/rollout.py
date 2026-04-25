"""
Training and rollout helpers for Gödel Env 2.0.

GodelEnv 2.0: The rollout system now supports two modes:
1. Strategy-patch mode: prompts include strategy context, completions are
   parsed as StrategyPatch proposals, rewards come from Governor decisions.
2. Legacy answer-improvement mode: backward-compatible with existing
   task-level training.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from godel_engine.async_utils import run_async
from godel_engine.environment import GodelEnvironment
from godel_engine.models import EditType, GodelAction, StrategyPatch
from godel_engine.symbolic_actions import (
    ACTION_DESCRIPTIONS,
    allowed_action_tokens,
    build_action_from_token,
    find_action_token,
)


logger = logging.getLogger("godel_env.rollout")


META_BLOCK_RE = re.compile(
    r"<godel_meta>\s*task_type=(?P<task_type>[^\n]+)\s*task_id=(?P<task_id>[^\n]+)\s*</godel_meta>",
    re.IGNORECASE,
)

TASK_BLOCK_RE = re.compile(
    r"TASK:\n(?P<task_prompt>.*?)\n\nCURRENT SOLUTION:\n",
    re.DOTALL,
)
CURRENT_SOLUTION_RE = re.compile(
    r"CURRENT SOLUTION:\n(?P<current_solution>.*?)\n\nLATEST FEEDBACK:\n",
    re.DOTALL,
)


def _extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def extract_task_prompt(prompt: str) -> str:
    match = TASK_BLOCK_RE.search(prompt)
    if not match:
        raise ValueError("Prompt is missing the TASK block.")
    return match.group("task_prompt").strip()


def extract_current_solution(prompt: str) -> str:
    match = CURRENT_SOLUTION_RE.search(prompt)
    if not match:
        raise ValueError("Prompt is missing the CURRENT SOLUTION block.")
    return match.group("current_solution").strip()


def _strategy_digest(strategy_text: str) -> str:
    lowered = strategy_text.lower()
    flags = [
        ("decompose", ["decompose", "break the problem"]),
        ("evidence", ["evidence", "support"]),
        ("counter", ["counterargument", "alternative hypothesis", "alternative"]),
        ("uncertainty", ["uncertainty", "confidence"]),
        ("verify", ["verify", "self-check", "verification"]),
        ("reflect", ["reflect", "revision", "lessons learned"]),
    ]
    parts = []
    for name, markers in flags:
        value = "yes" if any(marker in lowered for marker in markers) else "no"
        parts.append(f"{name}={value}")
    return ", ".join(parts)


def parse_completion_to_action(
    completion: str,
    *,
    task_prompt: str = "",
    task_type: str = "",
    current_solution: str = "",
    strategy_text: str = "",
) -> GodelAction:
    """Parse a model completion into an action object.

    GodelEnv 2.0: if the JSON contains an 'improved_strategy' key,
    we parse it as a StrategyPatch proposal.
    """
    blob = _extract_json_blob(completion)
    if blob is not None:
        try:
            data = json.loads(blob)

            # GodelEnv 2.0: detect strategy patch proposals
            if "improved_strategy" in data:
                patch = StrategyPatch(
                    improved_strategy=str(data["improved_strategy"]),
                    diff_description=str(data.get("diff_description", "")),
                    hypothesis=str(data.get("hypothesis", "")),
                    target_weaknesses=data.get("target_weaknesses", []),
                )
                return GodelAction(
                    solution=str(data.get("solution", data["improved_strategy"])),
                    edit_type=EditType.REWRITE,
                    strategy_note=str(data.get("strategy_note", "Strategy patch proposal")),
                    strategy_patch=patch,
                )

            # Legacy answer-improvement parsing
            solution = data.get("solution", completion)
            edit_type = str(data.get("edit_type", "rewrite")).upper()
            strategy_note = str(data.get("strategy_note", "RL rollout"))
            return GodelAction(
                solution=str(solution),
                edit_type=EditType[edit_type] if edit_type in EditType.__members__ else EditType.REWRITE,
                strategy_note=strategy_note,
            )
        except Exception:
            logger.debug("Falling back to raw completion after JSON parse failure.")

    action_token = find_action_token(completion)
    if action_token and task_type:
        return build_action_from_token(
            action_token,
            task_prompt=task_prompt,
            task_type=task_type,
            current_solution=current_solution,
            strategy_text=strategy_text,
        )

    code_match = re.search(r"```(?:python|json)?\s*\n?(.*?)```", completion, re.DOTALL)
    if code_match:
        return GodelAction(
            solution=code_match.group(1).strip(),
            edit_type=EditType.REWRITE,
            strategy_note="Extracted from fenced block",
        )

    return GodelAction(
        solution=completion.strip(),
        edit_type=EditType.REWRITE,
        strategy_note="Raw completion used as solution",
    )


def parse_prompt_metadata(prompt: str) -> dict[str, str]:
    match = META_BLOCK_RE.search(prompt)
    if not match:
        raise ValueError("Prompt is missing the <godel_meta> header.")
    return {
        "task_type": match.group("task_type").strip(),
        "task_id": match.group("task_id").strip(),
    }


def build_prompt(
    task_prompt: str,
    current_solution: str,
    rubric_feedback: str,
    task_type: str = "",
    task_id: str = "",
    strategy_text: str = "",
    strategy_id: str = "",
    downstream_scores: dict[str, float] | None = None,
    recent_failures: list[str] | None = None,
) -> str:
    """Build the model prompt used for training and inference.

    GodelEnv 2.0: prompts now include strategy context so the model
    can propose informed strategy patches.
    """
    meta = ""
    if task_type and task_id:
        meta = f"<godel_meta>\ntask_type={task_type}\ntask_id={task_id}\n</godel_meta>\n\n"

    # GodelEnv 2.0: strategy context block
    strategy_block = ""
    if strategy_text:
        strategy_block = (
            "CURRENT STRATEGY:\n"
            f"{strategy_text}\n\n"
        )
        if downstream_scores:
            scores_text = ", ".join(f"{k}: {v:.3f}" for k, v in downstream_scores.items())
            strategy_block += f"DOWNSTREAM SCORES: {scores_text}\n\n"
        if recent_failures:
            failures_text = "\n".join(f"- {f}" for f in recent_failures[-3:])
            strategy_block += f"RECENT FAILURES:\n{failures_text}\n\n"

    preferred_mode = (
        "For this task, prefer a strategy patch over a direct answer unless the current strategy is already stronger than the mutation you can propose.\n\n"
        if task_type == "strategy_optimization"
        else ""
    )
    action_tokens = allowed_action_tokens(task_type)
    action_menu = "\n".join(
        f"  - {token}: {ACTION_DESCRIPTIONS[token]}"
        for token in action_tokens
    )
    strategy_digest = _strategy_digest(strategy_text) if strategy_text else "n/a"
    score_digest = ", ".join(
        f"{name}={value:.2f}" for name, value in sorted((downstream_scores or {}).items())
    )
    failure_digest = "; ".join((recent_failures or [])[-2:])

    return (
        f"{meta}"
        "Choose exactly one action token for this GodelEnv state.\n"
        "ACTION MENU:\n"
        f"{action_menu}\n\n"
        f"{preferred_mode}"
        f"TASK:\n{task_prompt}\n\n"
        f"CURRENT SOLUTION:\n{current_solution}\n\n"
        f"STRATEGY DIGEST: {strategy_digest}\n"
        f"DOWNSTREAM SCORES: {score_digest or 'n/a'}\n"
        f"RECENT FAILURES: {failure_digest or 'n/a'}\n\n"
        f"LATEST FEEDBACK:\n{rubric_feedback}\n\n"
        "ACTION TOKEN:\n"
    )


def extract_reward_channels(step_result) -> Dict[str, float]:
    breakdown = step_result.reward_breakdown
    return {
        "task_score_delta": breakdown.task_score_delta,
        "format_compliance": breakdown.format_compliance,
        "step_cost": breakdown.step_cost,
        "anti_hack_penalty": breakdown.anti_hack_penalty,
        "process_reward": breakdown.process_reward,
        "total_reward": breakdown.total,
        "env_score": step_result.observation.total_score,
        # GodelEnv 2.0 channels
        "generalization_score": breakdown.generalization_score,
        "robustness_score": breakdown.robustness_score,
        "patch_quality": breakdown.patch_quality,
        "stability_score": breakdown.stability_score,
    }


def collect_local_prompt_dataset(
    *,
    num_prompts: int = 32,
    tasks: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Collect reset states from the local deterministic environment."""
    env = GodelEnvironment(seed=seed)
    task_names = tasks or list(env.tasks.keys())
    dataset: List[Dict[str, str]] = []

    async def _collect() -> None:
        task_instances: dict[str, list[str]] = {
            name: [item["id"] for item in getattr(env.tasks[name], "dataset", [])]
            for name in task_names
        }

        for index in range(num_prompts):
            task_type = task_names[index % len(task_names)]
            candidates = task_instances.get(task_type) or [""]
            task_id = candidates[index % len(candidates)] or None
            result = await env.reset(task_type=task_type, task_id=task_id, seed=seed + index)
            obs = result.observation
            dataset.append(
                {
                    "prompt": build_prompt(
                        task_prompt=obs.task_prompt,
                        current_solution=obs.current_solution,
                        rubric_feedback=obs.feedback_summary,
                        task_type=obs.task_type,
                        task_id=obs.task_id,
                        strategy_text=obs.current_strategy,
                        strategy_id=obs.strategy_id,
                        downstream_scores=obs.downstream_scores,
                        recent_failures=obs.recent_failures,
                    ),
                    "task_type": obs.task_type,
                    "task_id": obs.task_id,
                    "initial_score": f"{obs.total_score:.4f}",
                    "strategy_text": obs.current_strategy,
                    "strategy_id": obs.strategy_id,
                    "downstream_scores": dict(obs.downstream_scores),
                    "recent_failures": list(obs.recent_failures),
                }
            )

    run_async(_collect())
    return dataset


def make_local_grpo_rollout(max_new_tokens: int = 512):
    """Create a rollout function that scores completions on matching local task instances."""

    def rollout_func(prompts: List[str], trainer) -> Dict[str, Any]:
        tokenizer = trainer.processing_class
        torch = __import__("torch")

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {key: value.to(trainer.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            model_outputs = trainer.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lengths = [inputs["input_ids"].shape[1]] * len(prompts)
        else:
            prompt_lengths = attention_mask.sum(dim=1).tolist()

        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        completions: list[str] = []
        logits_all = model_outputs.logits

        for i, prompt in enumerate(prompts):
            meta = parse_prompt_metadata(prompt)
            action_ids = [
                tokenizer.convert_tokens_to_ids(token)
                for token in allowed_action_tokens(meta["task_type"])
            ]
            prompt_len = int(prompt_lengths[i])
            next_token_logits = logits_all[i, prompt_len - 1, :]
            full_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            filtered_logits = next_token_logits[action_ids] / 0.9
            filtered_probs = torch.softmax(filtered_logits, dim=-1)
            sampled_index = torch.multinomial(filtered_probs, num_samples=1).item()
            chosen_token_id = int(action_ids[sampled_index])

            completion_ids_out.append([chosen_token_id])
            logprobs_out.append([float(full_log_probs[chosen_token_id].item())])
            completions.append(
                tokenizer.decode([chosen_token_id], skip_special_tokens=False)
            )

        all_rewards = {
            "task_score_delta": [],
            "format_compliance": [],
            "anti_hack_penalty": [],
            "process_reward": [],
            "env_score": [],
            "patch_quality": [],
        }

        async def _step_all() -> None:
            for prompt, completion in zip(prompts, completions):
                env = GodelEnvironment()
                try:
                    meta = parse_prompt_metadata(prompt)
                    task_prompt = extract_task_prompt(prompt)
                    current_solution = extract_current_solution(prompt)
                    await env.reset(task_type=meta["task_type"], task_id=meta["task_id"])
                    action = parse_completion_to_action(
                        completion,
                        task_prompt=task_prompt,
                        task_type=meta["task_type"],
                        current_solution=current_solution,
                    )
                    result = await env.step(action)
                    channels = extract_reward_channels(result)
                    for key in all_rewards:
                        all_rewards[key].append(channels.get(key, 0.0))
                except Exception as exc:
                    logger.warning("Rollout step failed: %s", exc)
                    for key in all_rewards:
                        all_rewards[key].append(0.0)

        run_async(_step_all())

        return {
            "prompt_ids": inputs["input_ids"].tolist(),
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
            **all_rewards,
        }

    return rollout_func


def task_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return kwargs.get("task_score_delta", [0.0] * len(completions))


def format_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return kwargs.get("format_compliance", [0.0] * len(completions))


def guard_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return kwargs.get("anti_hack_penalty", [0.0] * len(completions))


def score_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    scores = kwargs.get("env_score", [0.0] * len(completions))
    return [score - 0.5 for score in scores]


def patch_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """GodelEnv 2.0: reward for strategy patch quality."""
    return kwargs.get("patch_quality", [0.0] * len(completions))


ALL_REWARD_FUNCS = [
    task_reward_func,
    format_reward_func,
    guard_reward_func,
    score_reward_func,
    patch_reward_func,
]
