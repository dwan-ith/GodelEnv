"""
Training support for Gödel Env 2.0.

GodelEnv 2.0: SFT warm-start now generates strategy patch examples
alongside answer completions. GRPO reward functions include patch_quality.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from godel_engine.async_utils import run_async

def extract_task_prompt(prompt: str) -> str:
    return prompt.split("TASK:\n", 1)[1].split("\n\nCURRENT SOLUTION:\n", 1)[0]


def build_supervised_examples(prompt_data: list[dict[str, str]]) -> list[dict[str, str]]:
    from godel_engine.heuristic_policy import build_heuristic_action

    examples: list[dict[str, str]] = []
    for item in prompt_data:
        task_prompt = extract_task_prompt(item["prompt"])
        action = build_heuristic_action(
            task_prompt,
            item["task_type"],
            strategy_text=item.get("strategy_text"),
        )
        completion = json.dumps(action.model_dump(mode="json"))
        examples.append({"text": item["prompt"] + completion})
    return examples


def load_tokenizer():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_tiny_model(tokenizer, *, max_length: int = 512):
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
    return GPT2LMHeadModel(config)


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
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_list(supervised_examples).map(
        tokenize, batched=True, remove_columns=["text"]
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
):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from godel_engine.rollout import (
        format_reward_func,
        guard_reward_func,
        make_local_grpo_rollout,
        patch_reward_func,
        score_reward_func,
        task_reward_func,
    )

    dataset = Dataset.from_list([{"prompt": item["prompt"]} for item in prompt_data])
    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
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
        rollout_func=make_local_grpo_rollout(max_new_tokens=max_new_tokens),
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
) -> dict[str, Any]:
    import torch

    from godel_engine.environment import GodelEnvironment
    from godel_engine.rollout import parse_completion_to_action

    records: list[dict[str, Any]] = []

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
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            action = parse_completion_to_action(completion)
            result = await env.step(action)

            record = {
                "task_type": item["task_type"],
                "task_id": item["task_id"],
                "reward": float(result.reward),
                "score": float(result.observation.total_score),
                "used_strategy_patch": bool(action.strategy_patch is not None),
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
        "task_means": task_means,
        "strategy_patch_rate": len(patch_records) / len(records),
        "patch_acceptance_rate": len(accepted_patches) / len(patch_records) if patch_records else 0.0,
        "mean_patch_improvement": mean_patch_improvement,
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

    # Right panel: recursive patch behavior
    ax = axes[2]
    recursive_labels = ["Patch rate", "Accept rate", "Mean patch delta"]
    baseline_recursive = [
        baseline_metrics.get("strategy_patch_rate", 0.0),
        baseline_metrics.get("patch_acceptance_rate", 0.0),
        baseline_metrics.get("mean_patch_improvement", 0.0),
    ]
    trained_recursive = [
        trained_metrics.get("strategy_patch_rate", 0.0),
        trained_metrics.get("patch_acceptance_rate", 0.0),
        trained_metrics.get("mean_patch_improvement", 0.0),
    ]
    x = range(len(recursive_labels))
    ax.bar([i - 0.18 for i in x], baseline_recursive, width=0.35, label="Baseline", color="#94a3b8")
    ax.bar([i + 0.18 for i in x], trained_recursive, width=0.35, label="Trained", color="#0f766e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(recursive_labels, fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Recursive Patch Metrics")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
