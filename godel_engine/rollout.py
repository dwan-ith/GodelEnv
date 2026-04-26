"""
Training and rollout helpers for Gödel Env.

GodelEnv: The rollout system now supports two modes:
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


def _attempt_structured_repair(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if '"solution"' in stripped and not stripped.startswith("{"):
        candidate = "{\n" + stripped
        if not candidate.rstrip().endswith("}"):
            candidate = candidate.rstrip() + "\n}"
        return candidate
    return None


def action_json_prefix(task_type: str) -> str:
    if task_type == "strategy_optimization":
        return '{\n  "improved_strategy": "'
    return '{\n  "solution": "'


def _extract_json_field(text: str, field_name: str) -> str | None:
    pattern = rf'"\s*,\s*"{re.escape(field_name)}"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return bytes(match.group("value"), "utf-8").decode("unicode_escape")


def _extract_json_list_field(text: str, field_name: str) -> list[str] | None:
    pattern = rf'"\s*,\s*"{re.escape(field_name)}"\s*:\s*\[(?P<value>.*?)\]'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    raw_value = "[" + match.group("value") + "]"
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return None
    return [str(item) for item in parsed]


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

    GodelEnv: The model generates full JSON completions, not compact tokens.
    This function parses JSON actions and extracts strategy patches when present.
    No symbolic action fallback - the model must learn to generate valid JSON.
    """
    blob = _extract_json_blob(completion) or _attempt_structured_repair(completion)
    if blob is not None:
        try:
            data = json.loads(blob)

            # GodelEnv: detect strategy patch proposals
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

            # Standard answer-improvement parsing
            solution = data.get("solution", completion)
            edit_type = str(data.get("edit_type", "rewrite")).upper()
            strategy_note = str(data.get("strategy_note", "Generated action"))
            return GodelAction(
                solution=str(solution),
                edit_type=EditType[edit_type] if edit_type in EditType.__members__ else EditType.REWRITE,
                strategy_note=strategy_note,
            )
        except Exception:
            logger.debug("JSON parse failed, using raw completion as solution.")

    prefix = action_json_prefix(task_type)
    if task_type == "strategy_optimization" and completion.startswith(prefix):
        remainder = completion[len(prefix):]
        field_boundary = re.search(r'"\s*,\s*"(diff_description|hypothesis|target_weaknesses|solution)"\s*:', remainder, re.DOTALL)
        improved_strategy = remainder[: field_boundary.start()].strip() if field_boundary else remainder.strip()
        tail = remainder[field_boundary.start() :] if field_boundary else ""
        solution = _extract_json_field(tail, "solution") or improved_strategy
        patch = StrategyPatch(
            improved_strategy=improved_strategy,
            diff_description=_extract_json_field(tail, "diff_description") or "Generated strategy update",
            hypothesis=_extract_json_field(tail, "hypothesis") or "A more explicit strategy should improve held-out performance.",
            target_weaknesses=_extract_json_list_field(tail, "target_weaknesses") or [],
        )
        return GodelAction(
            solution=solution,
            edit_type=EditType.REWRITE,
            strategy_note=_extract_json_field(tail, "strategy_note") or "Generated strategy patch",
            strategy_patch=patch,
        )

    if completion.startswith(prefix):
        remainder = completion[len(prefix):]
        field_boundary = re.search(r'"\s*,\s*"edit_type"\s*:', remainder, re.DOTALL)
        if field_boundary:
            solution_text = remainder[: field_boundary.start()].strip()
            edit_type = _extract_json_field(remainder[field_boundary.start() :], "edit_type") or "rewrite"
            strategy_note = _extract_json_field(remainder[field_boundary.start() :], "strategy_note") or (
                "Repaired prefixed JSON completion"
            )
            improved_strategy = _extract_json_field(remainder[field_boundary.start() :], "improved_strategy")
            if improved_strategy is not None:
                patch = StrategyPatch(
                    improved_strategy=improved_strategy,
                    diff_description=_extract_json_field(remainder[field_boundary.start() :], "diff_description") or "",
                    hypothesis=_extract_json_field(remainder[field_boundary.start() :], "hypothesis") or "",
                    target_weaknesses=_extract_json_list_field(remainder[field_boundary.start() :], "target_weaknesses")
                    or [],
                )
                return GodelAction(
                    solution=solution_text or improved_strategy,
                    edit_type=EditType[edit_type.upper()] if edit_type.upper() in EditType.__members__ else EditType.REWRITE,
                    strategy_note=strategy_note,
                    strategy_patch=patch,
                )
            return GodelAction(
                solution=solution_text,
                edit_type=EditType[edit_type.upper()] if edit_type.upper() in EditType.__members__ else EditType.REWRITE,
                strategy_note=strategy_note,
            )

        return GodelAction(
            solution=remainder.strip(),
            edit_type=EditType.REWRITE,
            strategy_note="Repaired prefixed JSON completion",
        )

    # Extract code blocks if present (common output format)
    code_match = re.search(r"```(?:python|json)?\s*\n?(.*?)```", completion, re.DOTALL)
    if code_match:
        return GodelAction(
            solution=code_match.group(1).strip(),
            edit_type=EditType.REWRITE,
            strategy_note="Extracted from fenced block",
        )

    # Use raw completion as solution (no symbolic token expansion)
    return GodelAction(
        solution=completion.strip(),
        edit_type=EditType.REWRITE,
        strategy_note="Raw completion used as solution",
    )


def classify_action_origin(action: GodelAction) -> str:
    if action.strategy_patch is not None:
        return "json_patch"
    if action.strategy_note == "Extracted from fenced block":
        return "code_block"
    if action.strategy_note == "Raw completion used as solution":
        return "raw_text"
    return "json_direct"


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

    GodelEnv: prompts instruct the model to generate full JSON actions,
    not compact action tokens. This enables genuine end-to-end learning.
    """
    meta = ""
    if task_type and task_id:
        meta = f"<godel_meta>\ntask_type={task_type}\ntask_id={task_id}\n</godel_meta>\n\n"

    # GodelEnv: strategy context block
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

    # Strategy optimization tasks should produce strategy patches
    if task_type == "strategy_optimization":
        output_instruction = (
            "Generate one JSON action that proposes an improved reasoning strategy.\n"
            "Required keys, in order: improved_strategy, diff_description, "
            "hypothesis, target_weaknesses, solution, edit_type, strategy_note.\n"
            "Use real task-specific content for every value. Do not copy field descriptions.\n"
        )
    else:
        output_instruction = (
            "Generate one JSON action.\n"
            "Required keys, in order: solution, edit_type, strategy_note.\n"
            "Use real task-specific content for every value. Do not copy field descriptions.\n"
        )

    strategy_digest = _strategy_digest(strategy_text) if strategy_text else "n/a"
    score_digest = ", ".join(
        f"{name}={value:.2f}" for name, value in sorted((downstream_scores or {}).items())
    )
    failure_digest = "; ".join((recent_failures or [])[-2:])

    return (
        f"{meta}"
        "You are an agent in GodelEnv, a recursive self-improvement environment.\n\n"
        f"{strategy_block}"
        f"TASK:\n{task_prompt}\n\n"
        f"CURRENT SOLUTION:\n{current_solution}\n\n"
        f"STRATEGY DIGEST: {strategy_digest}\n"
        f"DOWNSTREAM SCORES: {score_digest or 'n/a'}\n"
        f"RECENT FAILURES: {failure_digest or 'n/a'}\n\n"
        f"LATEST FEEDBACK:\n{rubric_feedback}\n\n"
        f"{output_instruction}\n"
        "Continue the JSON object below. Do not repeat the opening brace or first key.\n"
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
        # GodelEnv channels
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
) -> List[Dict[str, Any]]:
    """Collect reset states from the local deterministic environment."""
    # The dataset includes prompt text plus task metadata/reference objects used
    # for reference-grounded teacher traces during local training.
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
                    "reference": getattr(env.current_instance, "reference", None),
                }
            )

    run_async(_collect())
    return dataset


def make_local_grpo_rollout(max_new_tokens: int = 256):
    """
    Create a rollout function that generates full JSON completions.
    
    DEPRECATED: This function now behaves identically to make_freeform_grpo_rollout.
    The old symbolic-action approach has been removed in favor of genuine generation.
    Use make_freeform_grpo_rollout directly for clarity.
    """
    import warnings
    warnings.warn(
        "make_local_grpo_rollout is deprecated. Use make_freeform_grpo_rollout instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_freeform_grpo_rollout(max_new_tokens=max_new_tokens)


def make_freeform_grpo_rollout(max_new_tokens: int = 256):
    """
    Rollout where the model generates a full text completion (JSON or prose),
    then `parse_completion_to_action` maps it to a GodelAction.

    This is the path that actually trains a language model for self-improvement;
    :func:`make_local_grpo_rollout` only classifies a few special action tokens.
    """
    def rollout_func(prompts: List[str], trainer) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        tokenizer = trainer.processing_class

        prefixed_prompts = []
        prompt_prefixes = []
        for prompt in prompts:
            meta = parse_prompt_metadata(prompt)
            prefix = action_json_prefix(meta["task_type"])
            prompt_prefixes.append(prefix)
            prefixed_prompts.append(prompt + prefix)

        inputs = tokenizer(
            prefixed_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {key: value.to(trainer.model.device) for key, value in inputs.items()}

        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        completions: list[str] = []

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lengths = [inputs["input_ids"].shape[1]] * len(prefixed_prompts)
        else:
            prompt_lengths = attention_mask.sum(dim=1).tolist()

        for i, prompt in enumerate(prefixed_prompts):
            plen = int(prompt_lengths[i])
            row_input_ids = inputs["input_ids"][i : i + 1, :plen]
            row_mask = (
                attention_mask[i : i + 1, :plen]
                if attention_mask is not None
                else torch.ones_like(row_input_ids)
            )
            with torch.no_grad():
                out_ids = trainer.model.generate(
                    row_input_ids,
                    attention_mask=row_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_part = out_ids[0, plen:].tolist()
            completion_ids_out.append(gen_part)

            with torch.no_grad():
                model_out = trainer.model(
                    input_ids=out_ids,
                    attention_mask=torch.ones_like(out_ids, device=out_ids.device),
                )
            logits = model_out.logits[0]
            log_probs = F.log_softmax(logits, dim=-1)
            tok_lps: list[float] = []
            for t in range(max(plen, 1) - 1, int(out_ids.shape[1]) - 1):
                nxt = int(out_ids[0, t + 1].item())
                tok_lps.append(float(log_probs[t, nxt].item()))
            logprobs_out.append(tok_lps if tok_lps else [0.0])

            completions.append(prompt_prefixes[i] + tokenizer.decode(gen_part, skip_special_tokens=True))

        all_rewards: Dict[str, list] = {
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
                    origin = classify_action_origin(action)
                    if origin == "json_patch":
                        channels["format_compliance"] = channels.get("format_compliance", 0.0) + 0.08
                        channels["patch_quality"] = channels.get("patch_quality", 0.0) + 0.03
                    elif origin == "json_direct":
                        channels["format_compliance"] = channels.get("format_compliance", 0.0) + 0.05
                    elif origin == "code_block":
                        channels["format_compliance"] = channels.get("format_compliance", 0.0) + 0.02
                    for key in all_rewards:
                        all_rewards[key].append(channels.get(key, 0.0))
                except Exception as exc:
                    logger.warning("Freeform rollout step failed: %s", exc)
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
    """GodelEnv: reward for strategy patch quality."""
    return kwargs.get("patch_quality", [0.0] * len(completions))


ALL_REWARD_FUNCS = [
    task_reward_func,
    format_reward_func,
    guard_reward_func,
    score_reward_func,
    patch_reward_func,
]
