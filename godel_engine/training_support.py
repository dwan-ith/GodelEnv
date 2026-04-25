"""
Training support for Gödel Env 2.0.

GodelEnv 2.0 supports two training modes:

1. **Symbolic mode** (default, fast demo):
   - Model learns to emit one of ~3 compact action tokens
   - Tokens expand to handcrafted solutions via heuristic_policy
   - Evidence quality: PARTIAL - measures policy selection over hardcoded macros
   - Use for: quick smoke tests, CI, demo runs

2. **Freeform mode** (slower, real evidence):
   - Model learns to generate full JSON action completions
   - No expansion through hardcoded heuristics
   - Evidence quality: GENUINE - measures actual learned generation
   - Use for: real training runs, publishable results

The deterministic fallback evaluator uses handcrafted heuristics, so
"improvement" in deterministic+symbolic mode measures fit to the heuristic
world model, not genuine capability. For strong evidence, use:
  --generation-mode freeform --grading-mode auto --strategy-eval-mode auto
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from godel_engine.async_utils import run_async
from godel_engine.heuristic_policy import build_heuristic_action
from godel_engine.rollout import (
    extract_current_solution,
    extract_task_prompt,
    parse_completion_to_action,
)
from godel_engine.symbolic_actions import (
    ACTION_DESCRIPTIONS,
    allowed_action_tokens,
    preferred_action_token,
)


logger = logging.getLogger("godel_env.training")


def warn_evidence_quality(
    generation_mode: str,
    grading_mode: str,
    strategy_eval_mode: str,
) -> str:
    """
    Warn about evidence quality based on configuration.
    
    Returns a quality rating: "genuine", "partial", or "weak".
    """
    issues = []
    
    if generation_mode == "symbolic":
        issues.append(
            "generation_mode=symbolic: model selects from ~3 compact action tokens "
            "that expand to handcrafted solutions; not true end-to-end generation"
        )
    
    if grading_mode == "deterministic":
        issues.append(
            "grading_mode=deterministic: task grading uses handcrafted heuristics, "
            "not LLM evaluation"
        )
    
    if strategy_eval_mode == "deterministic":
        issues.append(
            "strategy_eval_mode=deterministic: held-out strategy evaluation uses "
            "handcrafted solvers; 'improvement' measures fit to heuristic world model"
        )
    
    if not issues:
        return "genuine"
    
    quality = "partial" if len(issues) <= 1 else "weak"
    
    warning_msg = (
        f"\n{'='*70}\n"
        f"EVIDENCE QUALITY: {quality.upper()}\n"
        f"{'='*70}\n"
        + "\n".join(f"  - {issue}" for issue in issues)
        + f"\n\nFor genuine evidence, use:\n"
        f"  --generation-mode freeform --grading-mode auto --strategy-eval-mode auto\n"
        f"{'='*70}\n"
    )
    warnings.warn(warning_msg, UserWarning, stacklevel=2)
    logger.warning(warning_msg)
    
    return quality


def build_supervised_examples(prompt_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Build SFT examples for SYMBOLIC mode (compact action tokens).
    
    WARNING: This trains the model to emit one of ~3 compact tokens that are
    then expanded via handcrafted heuristics. This is NOT end-to-end generation.
    Use build_supervised_examples_freeform() for genuine training evidence.
    """
    examples: list[dict[str, str]] = []
    for item in prompt_data:
        completion = preferred_action_token(item["task_type"])
        examples.append(
            {
                "prompt": item["prompt"],
                "completion": completion,
                "text": item["prompt"] + completion,
            }
        )
    return examples


def build_supervised_examples_freeform(
    prompt_data: list[dict[str, str]],
    *,
    include_patches: bool = True,
) -> list[dict[str, str]]:
    """
    Build SFT examples for FREEFORM mode (full JSON action generation).
    
    This trains the model to generate complete JSON actions, not compact tokens.
    The completions are real structured outputs, not expanded heuristics.
    """
    examples: list[dict[str, str]] = []
    
    for item in prompt_data:
        task_type = item["task_type"]
        task_prompt = extract_task_prompt(item["prompt"])
        current_solution = extract_current_solution(item["prompt"])
        strategy_text = item.get("strategy_text", "")
        
        # Build a demonstration action using heuristic policy (for SFT seed)
        # but the model will learn to generate this JSON format, not just tokens
        action = build_heuristic_action(
            task_prompt,
            task_type,
            strategy_text=strategy_text,
        )
        
        # Convert to JSON completion
        action_dict: dict[str, Any] = {
            "solution": action.solution,
            "edit_type": action.edit_type.value,
            "strategy_note": action.strategy_note or "Improved solution",
        }
        
        if include_patches and action.strategy_patch is not None:
            action_dict["improved_strategy"] = action.strategy_patch.improved_strategy
            action_dict["diff_description"] = action.strategy_patch.diff_description
            action_dict["hypothesis"] = action.strategy_patch.hypothesis
            action_dict["target_weaknesses"] = action.strategy_patch.target_weaknesses
        
        completion = json.dumps(action_dict, indent=2)
        
        examples.append({
            "prompt": item["prompt"],
            "completion": completion,
            "text": item["prompt"] + completion,
        })
    
    return examples


def load_tokenizer():
    from transformers import AutoTokenizer

    compact_tokens = list(ACTION_DESCRIPTIONS)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": compact_tokens})
    return tokenizer


def build_tiny_model(tokenizer, *, max_length: int = 512):
    """
    Build a tiny model for SYMBOLIC mode (compact token classification).
    
    This 2-layer model is sufficient for learning to select among ~3 action
    tokens, but is NOT capable of generating coherent JSON. Use
    build_freeform_model() for freeform generation training.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=128,
        n_layer=2,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    return model


def build_freeform_model(tokenizer, *, max_length: int = 1024):
    """
    Build a larger model for FREEFORM mode (actual JSON generation).
    
    This model has enough capacity to learn JSON structure and generate
    coherent action completions. It's ~10x larger than build_tiny_model().
    
    For production training, consider using a pre-trained model instead
    (e.g., a fine-tuned LLaMA or Mistral variant).
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=384,      # 3x larger embedding
        n_layer=6,       # 3x more layers
        n_head=6,        # More attention heads
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    return model


def run_sft(
    model,
    tokenizer,
    supervised_examples: list[dict[str, str]],
    *,
    output_dir: Path,
    max_steps: int = 6,
    batch_size: int = 1,
    max_length: int = 256,
    use_cpu: bool = True,
):
    from datasets import Dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    def tokenize(batch):
        input_ids_batch = []
        attention_masks = []
        labels_batch = []

        prompts = batch["prompt"]
        completions = batch["completion"]

        for prompt, completion in zip(prompts, completions):
            combined = prompt + completion
            encoded = tokenizer(
                combined,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            prompt_only = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            prompt_length = min(len(prompt_only["input_ids"]), max_length)
            labels = encoded["input_ids"][:]

            for idx in range(prompt_length):
                labels[idx] = -100
            for idx, mask in enumerate(encoded["attention_mask"]):
                if mask == 0:
                    labels[idx] = -100

            input_ids_batch.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks,
            "labels": labels_batch,
        }

    dataset = Dataset.from_list(supervised_examples).map(
        tokenize, batched=True, remove_columns=["prompt", "completion", "text"]
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        use_cpu=use_cpu,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    return trainer.state.log_history


def run_grpo(
    model,
    tokenizer,
    prompt_data: list[dict[str, str]],
    *,
    output_dir: Path,
    max_steps: int = 4,
    batch_size: int = 2,
    num_generations: int = 4,
    max_completion_length: int = 128,
    max_new_tokens: int = 128,
    use_cpu: bool = True,
    generation_mode: str = "symbolic",
):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from godel_engine.rollout import (
        format_reward_func,
        guard_reward_func,
        make_freeform_grpo_rollout,
        make_local_grpo_rollout,
        patch_reward_func,
        score_reward_func,
        task_reward_func,
    )

    if generation_mode == "freeform":
        rollout_fn = make_freeform_grpo_rollout(max_new_tokens=max_new_tokens)
        comp_len = max(max_completion_length, max_new_tokens)
    else:
        rollout_fn = make_local_grpo_rollout(max_new_tokens=max_new_tokens)
        comp_len = max_completion_length

    dataset = Dataset.from_list([{"prompt": item["prompt"]} for item in prompt_data])
    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=comp_len,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=use_cpu,
        # Temperature > 0 is essential: it ensures diverse completions
        # per prompt so reward variance is non-zero and GRPO can learn.
        generation_kwargs={
            "temperature": 0.9,
            "top_p": 0.95,
            "do_sample": True,
        },
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        rollout_func=rollout_fn,
        reward_funcs=[
            task_reward_func,
            format_reward_func,
            guard_reward_func,
            score_reward_func,
            patch_reward_func,
        ],
    )
    trainer.train()
    return trainer.state.log_history


def evaluate_model(
    model,
    tokenizer,
    prompt_data: list[dict[str, str]],
    *,
    max_new_tokens: int = 64,
    max_input_length: int = 256,
    policy_mode: str = "model",
    seed: int = 42,
) -> dict[str, Any]:
    import torch

    from godel_engine.environment import GodelEnvironment

    records: list[dict[str, Any]] = []
    chooser = random.Random(seed)

    async def _run() -> None:
        for item in prompt_data:
            env = GodelEnvironment()
            await env.reset(task_type=item["task_type"], task_id=item["task_id"])
            inputs = tokenizer(
                item["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
            )
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
            with torch.no_grad():
                output = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            prompt_length = int(inputs["attention_mask"].sum(dim=1).item())
            next_token_logits = output.logits[0, prompt_length - 1, :]
            action_ids = [
                tokenizer.convert_tokens_to_ids(token)
                for token in allowed_action_tokens(item["task_type"])
            ]
            if policy_mode == "random":
                chosen_token_id = int(chooser.choice(action_ids))
            else:
                filtered_logits = next_token_logits[action_ids]
                chosen_index = int(filtered_logits.argmax().item())
                chosen_token_id = int(action_ids[chosen_index])
            completion = tokenizer.decode([chosen_token_id], skip_special_tokens=False)
            action = parse_completion_to_action(
                completion,
                task_prompt=extract_task_prompt(item["prompt"]),
                task_type=item["task_type"],
                current_solution=extract_current_solution(item["prompt"]),
                strategy_text=item.get("strategy_text", ""),
            )
            result = await env.step(action)
            if action.strategy_patch is not None:
                action_origin = (
                    "compact_patch"
                    if action.strategy_note.startswith("Expanded from compact action token")
                    else "json_patch"
                )
            elif action.strategy_note.startswith("Expanded from compact action token"):
                action_origin = "compact_direct"
            elif action.strategy_note in {
                "Raw completion used as solution",
                "Extracted from fenced block",
            }:
                action_origin = "raw_direct"
            else:
                action_origin = "json_direct"

            record = {
                "task_type": item["task_type"],
                "task_id": item["task_id"],
                "reward": float(result.reward),
                "score": float(result.observation.total_score),
                "used_strategy_patch": bool(action.strategy_patch is not None),
                "action_origin": action_origin,
            }

            # GodelEnv 2.0: log patch decision if present
            if result.patch_decision is not None:
                record["patch_accepted"] = result.patch_decision.accepted
                record["patch_improvement"] = result.patch_decision.improvement

            records.append(record)

    run_async(_run())

    if not records:
        return {"mean_reward": 0.0, "mean_score": 0.0, "episodes": []}

    mean_reward = sum(record["reward"] for record in records) / len(records)
    mean_score = sum(record["score"] for record in records) / len(records)
    task_buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        task_buckets[record["task_type"]].append(record["score"])
    task_means = {
        task_type: sum(scores) / len(scores)
        for task_type, scores in task_buckets.items()
        if scores
    }

    structured_action_rate = sum(
        1
        for record in records
        if record.get("action_origin", "").startswith(("compact_", "json_"))
    ) / len(records)
    patch_records = [record for record in records if record.get("used_strategy_patch")]
    accepted_patches = [record for record in patch_records if record.get("patch_accepted")]
    mean_patch_improvement = (
        sum(float(record.get("patch_improvement", 0.0)) for record in patch_records) / len(patch_records)
        if patch_records
        else 0.0
    )
    return {
        "mean_reward": mean_reward,
        "mean_score": mean_score,
        "policy_mode": policy_mode,
        "task_means": task_means,
        "structured_action_rate": structured_action_rate,
        "strategy_patch_rate": len(patch_records) / len(records),
        "patch_acceptance_rate": len(accepted_patches) / len(patch_records) if patch_records else 0.0,
        "mean_patch_improvement": mean_patch_improvement,
        "generation_mode": "symbolic",  # Document that this is symbolic mode
        "episodes": records,
    }


def evaluate_model_freeform(
    model,
    tokenizer,
    prompt_data: list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
    max_input_length: int = 512,
    policy_mode: str = "model",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate model in FREEFORM mode (actual JSON generation).
    
    Unlike evaluate_model() which picks from compact action tokens, this
    generates full JSON completions and parses them. This provides genuine
    evidence of learned generation capability.
    """
    import torch

    from godel_engine.environment import GodelEnvironment

    records: list[dict[str, Any]] = []
    chooser = random.Random(seed)

    async def _run() -> None:
        for item in prompt_data:
            env = GodelEnvironment()
            await env.reset(task_type=item["task_type"], task_id=item["task_id"])
            
            inputs = tokenizer(
                item["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
            )
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
            
            if policy_mode == "random":
                # Random baseline: generate with high temperature
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=2.0,  # Very high for random-ish output
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            else:
                # Model policy: generate with moderate temperature
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            
            # Extract only the generated part
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, prompt_length:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Parse the completion into an action
            action = parse_completion_to_action(
                completion,
                task_prompt=extract_task_prompt(item["prompt"]),
                task_type=item["task_type"],
                current_solution=extract_current_solution(item["prompt"]),
                strategy_text=item.get("strategy_text", ""),
            )
            
            result = await env.step(action)
            
            # Determine action origin (was it parsed from JSON or raw text?)
            if action.strategy_patch is not None:
                action_origin = "freeform_json_patch"
            elif action.strategy_note.startswith("Expanded from compact action token"):
                # Model generated a compact token - not truly freeform
                action_origin = "freeform_fell_back_to_symbolic"
            elif action.strategy_note in {
                "Raw completion used as solution",
                "Extracted from fenced block",
            }:
                action_origin = "freeform_raw_text"
            else:
                action_origin = "freeform_json_direct"

            record = {
                "task_type": item["task_type"],
                "task_id": item["task_id"],
                "reward": float(result.reward),
                "score": float(result.observation.total_score),
                "used_strategy_patch": bool(action.strategy_patch is not None),
                "action_origin": action_origin,
                "completion_length": len(completion),
            }

            if result.patch_decision is not None:
                record["patch_accepted"] = result.patch_decision.accepted
                record["patch_improvement"] = result.patch_decision.improvement

            records.append(record)

    run_async(_run())

    if not records:
        return {"mean_reward": 0.0, "mean_score": 0.0, "episodes": [], "generation_mode": "freeform"}

    mean_reward = sum(record["reward"] for record in records) / len(records)
    mean_score = sum(record["score"] for record in records) / len(records)
    task_buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        task_buckets[record["task_type"]].append(record["score"])
    task_means = {
        task_type: sum(scores) / len(scores)
        for task_type, scores in task_buckets.items()
        if scores
    }

    # Track how often the model generated valid JSON vs fell back
    json_action_rate = sum(
        1
        for record in records
        if "json" in record.get("action_origin", "")
    ) / len(records)
    
    symbolic_fallback_rate = sum(
        1
        for record in records
        if "symbolic" in record.get("action_origin", "")
    ) / len(records)
    
    patch_records = [record for record in records if record.get("used_strategy_patch")]
    accepted_patches = [record for record in patch_records if record.get("patch_accepted")]
    mean_patch_improvement = (
        sum(float(record.get("patch_improvement", 0.0)) for record in patch_records) / len(patch_records)
        if patch_records
        else 0.0
    )
    
    return {
        "mean_reward": mean_reward,
        "mean_score": mean_score,
        "policy_mode": policy_mode,
        "task_means": task_means,
        "json_action_rate": json_action_rate,
        "symbolic_fallback_rate": symbolic_fallback_rate,
        "strategy_patch_rate": len(patch_records) / len(records),
        "patch_acceptance_rate": len(accepted_patches) / len(patch_records) if patch_records else 0.0,
        "mean_patch_improvement": mean_patch_improvement,
        "generation_mode": "freeform",
        "episodes": records,
    }


def plot_training_curves(
    sft_logs: list[dict[str, Any]],
    grpo_logs: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sft_steps = [item["step"] for item in sft_logs if "loss" in item]
    sft_losses = [item["loss"] for item in sft_logs if "loss" in item]
    # TRL GRPOTrainer logs reward under "reward" key
    reward_steps = [item["step"] for item in grpo_logs if "reward" in item]
    rewards = [item["reward"] for item in grpo_logs if "reward" in item]
    # Also collect per-channel rewards when available
    task_rewards = [item.get("rewards/task_reward_func/mean", None) for item in grpo_logs if "reward" in item]
    format_rewards = [item.get("rewards/format_reward_func/mean", None) for item in grpo_logs if "reward" in item]

    loss_path = output_dir / "loss_curve.png"
    reward_path = output_dir / "reward_curve.png"

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(7, 4))
    if sft_steps:
        ax.plot(sft_steps, sft_losses, marker="o", color="#1d4ed8", linewidth=2)
    else:
        ax.text(0.5, 0.5, "No SFT logs", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("SFT Step")
    ax.set_ylabel("Loss")
    ax.set_title("GodelEnv — SFT Warm-Start Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    # ── Reward curve ──
    fig, ax = plt.subplots(figsize=(7, 4))
    if reward_steps:
        ax.plot(reward_steps, rewards, marker="o", color="#059669", linewidth=2, label="Total reward")
        # Plot per-channel if available
        if any(v is not None for v in task_rewards):
            ax.plot(reward_steps, [v or 0 for v in task_rewards], marker="s", color="#7c3aed",
                    linewidth=1.5, alpha=0.7, label="Task score delta")
        if any(v is not None for v in format_rewards):
            ax.plot(reward_steps, [v or 0 for v in format_rewards], marker="^", color="#ea580c",
                    linewidth=1.5, alpha=0.7, label="Format compliance")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GRPO logs", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("GodelEnv — GRPO Reinforcement Learning Reward")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reward_path, dpi=160)
    plt.close(fig)

    return {"loss_curve": loss_path, "reward_curve": reward_path}


def plot_before_after(
    baseline_metrics: dict[str, Any],
    trained_metrics: dict[str, Any],
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "before_after.png"

    labels = ["Mean Reward", "Mean Score"]
    baseline_values = [
        baseline_metrics.get("mean_reward", 0.0),
        baseline_metrics.get("mean_score", 0.0),
    ]
    trained_values = [
        trained_metrics.get("mean_reward", 0.0),
        trained_metrics.get("mean_score", 0.0),
    ]

    # Also compute per-task breakdown if episodes data is available
    baseline_episodes = baseline_metrics.get("episodes", [])
    trained_episodes = trained_metrics.get("episodes", [])

    positions = range(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left panel: aggregate before/after
    ax = axes[0]
    ax.bar([p - 0.18 for p in positions], baseline_values, width=0.35, label="Baseline", color="#94a3b8")
    ax.bar([p + 0.18 for p in positions], trained_values, width=0.35, label="Trained", color="#7c3aed")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("GodelEnv — Before / After Training")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Middle panel: per-task score comparison
    ax = axes[1]
    if baseline_episodes and trained_episodes:
        from collections import defaultdict
        baseline_by_task: dict[str, list[float]] = defaultdict(list)
        trained_by_task: dict[str, list[float]] = defaultdict(list)
        for ep in baseline_episodes:
            baseline_by_task[ep["task_type"]].append(ep["score"])
        for ep in trained_episodes:
            trained_by_task[ep["task_type"]].append(ep["score"])
        tasks = sorted(set(baseline_by_task) | set(trained_by_task))
        b_scores = [sum(baseline_by_task[t]) / max(len(baseline_by_task[t]), 1) for t in tasks]
        t_scores = [sum(trained_by_task[t]) / max(len(trained_by_task[t]), 1) for t in tasks]
        x = range(len(tasks))
        ax.bar([i - 0.18 for i in x], b_scores, width=0.35, label="Baseline", color="#94a3b8")
        ax.bar([i + 0.18 for i in x], t_scores, width=0.35, label="Trained", color="#7c3aed")
        ax.set_xticks(list(x))
        ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title("Per-Task Score Breakdown")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No per-task data", ha="center", va="center", transform=ax.transAxes)

    # Right panel: policy behavior
    ax = axes[2]
    behavior_labels = ["Structured rate", "Strategy score", "Patch rate"]
    baseline_recursive = [
        baseline_metrics.get("structured_action_rate", 0.0),
        baseline_metrics.get("task_means", {}).get("strategy_optimization", 0.0),
        baseline_metrics.get("strategy_patch_rate", 0.0),
    ]
    trained_recursive = [
        trained_metrics.get("structured_action_rate", 0.0),
        trained_metrics.get("task_means", {}).get("strategy_optimization", 0.0),
        trained_metrics.get("strategy_patch_rate", 0.0),
    ]
    x = range(len(behavior_labels))
    ax.bar([i - 0.18 for i in x], baseline_recursive, width=0.35, label="Baseline", color="#94a3b8")
    ax.bar([i + 0.18 for i in x], trained_recursive, width=0.35, label="Trained", color="#0f766e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(behavior_labels, fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Policy Behavior")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
