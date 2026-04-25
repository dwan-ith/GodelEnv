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


logger = logging.getLogger("godel_env.rollout")


META_BLOCK_RE = re.compile(
    r"<godel_meta>\s*task_type=(?P<task_type>[^\n]+)\s*task_id=(?P<task_id>[^\n]+)\s*</godel_meta>",
    re.IGNORECASE,
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


def parse_completion_to_action(completion: str) -> GodelAction:
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

    return (
        f"{meta}"
        "You are an agent inside GodelEnv. Your goal is recursive self-improvement.\n"
        "You can either:\n"
        "  A) Propose a strategy patch (include 'improved_strategy' in your JSON)\n"
        "  B) Rewrite the solution directly (include 'solution' in your JSON)\n\n"
        "Return valid JSON with the appropriate keys.\n\n"
        f"{preferred_mode}"
        f"{strategy_block}"
        f"TASK:\n{task_prompt}\n\n"
        f"CURRENT SOLUTION:\n{current_solution}\n\n"
        f"LATEST FEEDBACK:\n{rubric_feedback}\n"
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
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # IMPORTANT: prompt length is per-sample (because of padding).
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lengths = [inputs["input_ids"].shape[1]] * len(prompts)
        else:
            prompt_lengths = attention_mask.sum(dim=1).tolist()

        completion_id_tensors: list[Any] = []
        completions: list[str] = []
        for i, prompt_len in enumerate(prompt_lengths):
            comp_ids = outputs[i, prompt_len:]
            completion_id_tensors.append(comp_ids)
            completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True))

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
                    await env.reset(task_type=meta["task_type"], task_id=meta["task_id"])
                    action = parse_completion_to_action(completion)
                    result = await env.step(action)
                    channels = extract_reward_channels(result)
                    for key in all_rewards:
                        all_rewards[key].append(channels.get(key, 0.0))
                except Exception as exc:
                    logger.warning("Rollout step failed: %s", exc)
                    for key in all_rewards:
                        all_rewards[key].append(0.0)

        run_async(_step_all())

        # Compute per-token logprobs for each completion
        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []

        with torch.no_grad():
            model_outputs = trainer.model(
                input_ids=outputs,
                attention_mask=(outputs != tokenizer.pad_token_id).long()
                if tokenizer.pad_token_id is not None
                else torch.ones_like(outputs),
            )
            logits_all = model_outputs.logits  # (B, T, V)
            log_probs_all = torch.log_softmax(logits_all, dim=-1)

            for i, prompt_len in enumerate(prompt_lengths):
                comp_ids = completion_id_tensors[i]
                completion_ids_out.append(comp_ids.tolist())
                if comp_ids.numel() == 0:
                    logprobs_out.append([])
                    continue

                token_positions = torch.arange(prompt_len, prompt_len + comp_ids.shape[0], device=outputs.device)
                logits_positions = token_positions - 1
                chosen_log_probs = log_probs_all[i, logits_positions, :].gather(
                    1, comp_ids.unsqueeze(-1)
                ).squeeze(-1)
                logprobs_out.append(chosen_log_probs.tolist())

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
