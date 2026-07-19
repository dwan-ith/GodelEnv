"""
Training support for Gödel Env

GodelEnv trains models to generate full JSON action completions,
enabling end-to-end learning of self-improvement.

The model learns to:
1. Generate valid JSON actions with solution, edit_type, and strategy_note
2. Propose strategy patches with improved_strategy, hypothesis, and target_weaknesses
3. Propose bounded environment patches from immutable verified task IDs
4. Demonstrate proposed strategies through concrete task solutions

Evidence quality depends on:
- grading_mode: 'auto' (uses LLM) vs 'deterministic' (uses heuristic graders)
- strategy_eval_mode: 'auto' (uses LLM) vs 'deterministic' (uses heuristic solvers)

Deterministic verifiers provide reproducible RLVR evidence. API-backed evaluation
is a separate integration mode and must report provider sources explicitly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from godel_engine.async_utils import run_async
from godel_engine.deterministic_solver import build_reference_action
from godel_engine.rollout import (
    action_json_prefix,
    classify_action_origin,
    extract_current_solution,
    extract_task_prompt,
    format_policy_prompt,
    inspect_action_completion,
    parse_completion_to_action,
    reconstruct_action_completion,
)


logger = logging.getLogger("godel_env.training")


def _cached_model_source(model_name: str) -> str | None:
    """Resolve a Hub model to its local snapshot without making a network request."""
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate.resolve())
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return None


def _ensure_tokenizer_capacity(model, tokenizer) -> None:
    """Grow embeddings only when required; shrinking marks huge PEFT tensors dirty."""
    current_size = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) > current_size:
        model.resize_token_embeddings(len(tokenizer))


def _apply_lora_if_enabled(model):
    """Use adapter training by default so GRPO fits hackathon-class GPUs."""
    if os.getenv("GODEL_FULL_FINETUNE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return model
    from peft import LoraConfig, get_peft_model

    linear_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    target_modules = [name for name in preferred if name in linear_names]
    if not target_modules:
        raise RuntimeError(
            "Could not identify LoRA target modules for the selected base model. "
            "Set GODEL_FULL_FINETUNE=1 only if full fine-tuning is intentional."
        )
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Enabled LoRA: %s/%s trainable parameters (%.3f%%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return model


def build_supervised_examples(prompt_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Build SFT examples for JSON action generation.
    """
    return build_supervised_examples_freeform(prompt_data)


def build_supervised_examples_freeform(
    prompt_data: list[dict[str, str]],
    *,
    include_patches: bool = True,
) -> list[dict[str, str]]:
    """
    Build SFT examples for full JSON action generation.

    These teacher traces are reference-grounded deterministic actions, so the
    model learns the real action schema instead of a tiny macro vocabulary.
    """
    examples: list[dict[str, str]] = []
    
    for item in prompt_data:
        task_type = item["task_type"]
        task_prompt = extract_task_prompt(item["prompt"])
        current_solution = extract_current_solution(item["prompt"])
        strategy_text = item.get("strategy_text", "")
        
        action = build_reference_action(
            task_prompt=task_prompt,
            task_type=task_type,
            strategy_text=strategy_text,
            recent_failures=item.get("recent_failures", []),
            downstream_scores=item.get("downstream_scores", {}),
            reference=item.get("reference"),
        )
        
        # Convert to JSON completion
        if include_patches and action.strategy_patch is not None:
            action_dict = {
                "improved_strategy": action.strategy_patch.improved_strategy,
                "diff_description": action.strategy_patch.diff_description,
                "hypothesis": action.strategy_patch.hypothesis,
                "target_weaknesses": action.strategy_patch.target_weaknesses,
            }
            if action.environment_patch is not None:
                action_dict["environment_patch"] = action.environment_patch.model_dump(mode="json")
        else:
            action_dict = {
                "solution": action.solution,
                "edit_type": action.edit_type.value,
                "strategy_note": action.strategy_note or "Improved solution",
            }
        
        full_completion = json.dumps(action_dict, indent=2, ensure_ascii=False)
        prefix = action_json_prefix(task_type)
        if full_completion.startswith(prefix):
            completion = full_completion[len(prefix):]
        else:
            completion = full_completion
        
        examples.append({
            "prompt": item["prompt"],
            "completion": completion,
            "task_type": task_type,
            "text": item["prompt"] + full_completion,
        })
    
    return examples


def oversample_recursive_examples(
    examples: list[dict[str, str]],
    *,
    multiplier: int = 4,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Increase dual-mutation schema exposure while retaining every anchor row."""
    if multiplier < 1:
        raise ValueError("multiplier must be at least 1")
    recursive = [row for row in examples if row.get("task_type") == "strategy_optimization"]
    rows = list(examples)
    for _ in range(multiplier - 1):
        rows.extend(dict(row) for row in recursive)
    random.Random(seed).shuffle(rows)
    return rows


def load_tokenizer():
    from transformers import AutoTokenizer

    model_name = os.getenv("GODEL_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    local_source = _cached_model_source(model_name)
    if local_source is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            local_source,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        logger.info("Loaded tokenizer from local cache: %s", model_name)
    else:
        logger.info("No complete local tokenizer snapshot for %s; downloading", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=False)
        logger.info("Downloaded tokenizer: %s", model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_tiny_model(tokenizer, *, max_length: int = 1536):
    """
    Build a small model for quick training demos.
    
    This is a smaller version of build_freeform_model() for faster iteration.
    For production training, use a larger pre-trained model.
    """
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    model_name = os.getenv("GODEL_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    local_source = _cached_model_source(model_name)
    try:
        if local_source is None:
            raise FileNotFoundError(f"no complete local snapshot for {model_name}")
        model = AutoModelForCausalLM.from_pretrained(local_source, local_files_only=True)
        logger.info("Loaded pretrained causal LM from local cache: %s", model_name)
    except Exception as local_exc:
        logger.warning("Local model load failed for %s: %s", model_name, local_exc)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Downloaded pretrained causal LM: %s", model_name)
        except Exception as remote_exc:
            if os.getenv("GODEL_ALLOW_RANDOM_MODEL", "0").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                raise RuntimeError(
                    f"Could not load pretrained model {model_name}. Random initialization is "
                    "disabled because it invalidates training evidence. Set "
                    "GODEL_ALLOW_RANDOM_MODEL=1 only for architecture smoke tests."
                ) from remote_exc
            logger.warning(
                "Falling back to randomly initialized tiny model because pretrained load failed for %s: %s",
                model_name,
                remote_exc,
            )
            config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_positions=max_length,
                n_ctx=max_length,
                n_embd=256,
                n_layer=4,
                n_head=4,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            model = GPT2LMHeadModel(config)
            _ensure_tokenizer_capacity(model, tokenizer)
            return model

    if getattr(model.config, "n_positions", max_length) < max_length:
        logger.warning(
            "Requested max_length=%s exceeds pretrained context window=%s; using model limit.",
            max_length,
            getattr(model.config, "n_positions", None),
        )
    _ensure_tokenizer_capacity(model, tokenizer)
    return _apply_lora_if_enabled(model)


def build_freeform_model(tokenizer, *, max_length: int = 2048):
    """
    Build a larger model for FREEFORM mode (actual JSON generation).
    
    This model has enough capacity to learn JSON structure and generate
    coherent action completions. It's ~10x larger than build_tiny_model().
    
    For production training, consider using a pre-trained model instead
    (e.g., a fine-tuned LLaMA or Mistral variant).
    """
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    model_name = os.getenv("GODEL_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    local_source = _cached_model_source(model_name)
    try:
        if local_source is None:
            raise FileNotFoundError(f"no complete local snapshot for {model_name}")
        model = AutoModelForCausalLM.from_pretrained(local_source, local_files_only=True)
        logger.info("Loaded pretrained causal LM from local cache: %s", model_name)
    except Exception as local_exc:
        logger.warning("Local model load failed for %s: %s", model_name, local_exc)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Downloaded pretrained causal LM: %s", model_name)
        except Exception as remote_exc:
            if os.getenv("GODEL_ALLOW_RANDOM_MODEL", "0").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                raise RuntimeError(
                    f"Could not load pretrained model {model_name}. Random initialization is "
                    "disabled because it invalidates training evidence. Set "
                    "GODEL_ALLOW_RANDOM_MODEL=1 only for architecture smoke tests."
                ) from remote_exc
            logger.warning(
                "Falling back to randomly initialized freeform model because pretrained load failed for %s: %s",
                model_name,
                remote_exc,
            )
            config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_positions=max_length,
                n_ctx=max_length,
                n_embd=384,
                n_layer=6,
                n_head=6,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            model = GPT2LMHeadModel(config)
            _ensure_tokenizer_capacity(model, tokenizer)
            return model

    if getattr(model.config, "n_positions", max_length) < max_length:
        logger.warning(
            "Requested max_length=%s exceeds pretrained context window=%s; using model limit.",
            max_length,
            getattr(model.config, "n_positions", None),
        )
    _ensure_tokenizer_capacity(model, tokenizer)
    return _apply_lora_if_enabled(model)


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
    learning_rate: float = 5e-5,
):
    import torch

    from datasets import Dataset
    from transformers import (
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    def tokenize(batch):
        input_ids_batch = []
        attention_masks = []
        labels_batch = []

        prompts = batch["prompt"]
        completions = batch["completion"]
        task_types = batch["task_type"]

        for prompt, completion, task_type in zip(
            prompts, completions, task_types, strict=True
        ):
            model_prompt = format_policy_prompt(tokenizer, prompt, task_type)
            completion_with_eos = completion + (tokenizer.eos_token or "")
            completion_ids = tokenizer(
                completion_with_eos,
                add_special_tokens=False,
            )["input_ids"]
            if len(completion_ids) >= max_length:
                raise ValueError(
                    "SFT completion does not fit in max_length; increase the training context"
                )
            prompt_encoded = tokenizer(
                model_prompt,
                truncation=True,
                max_length=max_length - len(completion_ids),
                add_special_tokens=True,
            )
            prompt_ids = prompt_encoded["input_ids"]
            input_ids = prompt_ids + completion_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + completion_ids[:]
            padding = max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding)
            attention_mask.extend([0] * padding)
            labels.extend([-100] * padding)

            input_ids_batch.append(input_ids)
            attention_masks.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks,
            "labels": labels_batch,
        }

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        dataset = Dataset.from_list(supervised_examples).map(
            tokenize,
            batched=True,
            remove_columns=["prompt", "completion", "task_type", "text"],
        )
    finally:
        tokenizer.padding_side = original_padding_side
    if any(not any(label != -100 for label in row) for row in dataset["labels"]):
        raise ValueError(
            "At least one SFT example lost its entire completion to truncation. "
            "Increase max_length or shorten the environment prompt."
        )
    use_bf16 = bool(not use_cpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        use_cpu=use_cpu,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        seed=42,
        data_seed=42,
        bf16=use_bf16,
        fp16=bool(not use_cpu and torch.cuda.is_available() and not use_bf16),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    return trainer.state.log_history


def build_adaptive_repair_examples(
    supervised_examples: list[dict[str, str]],
    comparison: dict[str, Any],
    *,
    oversample: int = 3,
    tolerance: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[str]]:
    """Replay all anchors while emphasizing task families that regressed."""
    if oversample < 1:
        raise ValueError("oversample must be at least 1")
    weak_tasks = sorted(
        task_type
        for task_type, delta in comparison.get("per_task_score_delta", {}).items()
        if float(delta) < -tolerance
    )
    if not weak_tasks:
        return [], []

    rows = list(supervised_examples)
    for example in supervised_examples:
        if example["task_type"] in weak_tasks:
            rows.extend([example] * (oversample - 1))
    random.Random(seed).shuffle(rows)
    return rows, weak_tasks


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
    generation_mode: str = "freeform",
    recursive_oversample: int = 3,
    learning_rate: float = 2e-6,
):
    """
    Run GRPO training with reward functions.
    
    Compatible with TRL 0.15+ (uses reward_funcs for custom reward scoring).
    """
    import inspect
    import torch

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from godel_engine.training_rewards import EnvironmentRewardSuite

    comp_len = max(max_completion_length, max_new_tokens)

    if recursive_oversample < 1:
        raise ValueError("recursive_oversample must be at least 1")
    rows: list[dict[str, Any]] = []
    for item in prompt_data:
        repeats = recursive_oversample if item["task_type"] == "strategy_optimization" else 1
        for repeat in range(repeats):
            rows.append({
                "prompt": format_policy_prompt(tokenizer, item["prompt"], item["task_type"]),
                "env_prompt": item["prompt"],
                "task_type": item["task_type"],
                "task_id": item["task_id"],
                "initial_score": float(item.get("initial_score", 0.0)),
                "split": item.get("split", "train"),
                "curriculum_repeat": repeat,
            })
    random.Random(42).shuffle(rows)
    dataset = Dataset.from_list(rows)
    reward_suite = EnvironmentRewardSuite(seed=42)
    
    # Build config with parameters compatible across TRL versions
    config_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "num_generations": num_generations,
        "max_completion_length": comp_len,
        "max_steps": max_steps,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "learning_rate": learning_rate,
        "warmup_ratio": 0.1,
        "seed": 42,
        "data_seed": 42,
    }
    
    # Add use_cpu only if supported (older TRL versions)
    grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
    if "use_cpu" in grpo_config_params:
        config_kwargs["use_cpu"] = use_cpu
    if not use_cpu and torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
        if "bf16" in grpo_config_params:
            config_kwargs["bf16"] = use_bf16
        if "fp16" in grpo_config_params:
            config_kwargs["fp16"] = not use_bf16
    
    optional_config = {
        "temperature": 0.8,
        "top_p": 0.95,
        "beta": 0.02,
        "scale_rewards": "batch",
        "loss_type": "dr_grpo",
        "mask_truncated_completions": True,
        "remove_unused_columns": False,
        "reward_weights": reward_suite.weights,
    }
    for name, value in optional_config.items():
        if name in grpo_config_params:
            config_kwargs[name] = value
    
    args = GRPOConfig(**config_kwargs)
    
    # Build trainer with parameters compatible across TRL versions
    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "reward_funcs": reward_suite.reward_functions(),
    }
    
    # Handle tokenizer/processing_class parameter name change
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    return trainer.state.log_history


def evaluate_model(
    model,
    tokenizer,
    prompt_data: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    max_input_length: int = 512,
    policy_mode: str = "model",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a model by generating full JSON actions.
    """
    return evaluate_model_freeform(
        model,
        tokenizer,
        prompt_data,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
        policy_mode=policy_mode,
        seed=seed,
    )


def evaluate_model_freeform(
    model,
    tokenizer,
    prompt_data: list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
    max_input_length: int = 512,
    policy_mode: str = "model",
    seed: int = 42,
    generation_batch_size: int = 4,
    base_fallback_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a policy on fixed prompts using strict JSON and environment execution."""
    import torch

    from godel_engine.training_rewards import EnvironmentRewardSuite

    from godel_engine.adapter_routing import AdapterRoutingPolicy

    base_fallback_tasks = set(base_fallback_tasks or ())
    routing_policy = AdapterRoutingPolicy(frozenset(base_fallback_tasks))
    if base_fallback_tasks:
        if not hasattr(model, "disable_adapter"):
            raise TypeError("Base fallback routing requires a PEFT model with disable_adapter()")
        generation_batch_size = 1

    generated_texts: list[str] = []
    generated_token_lengths: list[int] = []
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    try:
        for batch_start in range(0, len(prompt_data), generation_batch_size):
            batch = prompt_data[batch_start : batch_start + generation_batch_size]
            generation_prompts = [
                format_policy_prompt(tokenizer, item["prompt"], item["task_type"])
                for item in batch
            ]
            context_limit = int(
                getattr(model.config, "max_position_embeddings", 0)
                or getattr(model.config, "n_positions", 0)
                or max_input_length + max_new_tokens
            )
            input_limit = min(
                max_input_length,
                max(32, context_limit - max_new_tokens),
            )
            inputs = tokenizer(
                generation_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=input_limit,
            )
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
            generation_kwargs: dict[str, Any] = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if policy_mode == "sample":
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.95,
                    }
                )
            else:
                generation_kwargs["do_sample"] = False

            adapter_context = routing_policy.model_context(model, batch[0]["task_type"])
            with adapter_context, torch.no_grad():
                output_ids = model.generate(**generation_kwargs)
            prompt_length = inputs["input_ids"].shape[1]
            for sequence in output_ids:
                generated_ids = sequence[prompt_length:]
                generated_texts.append(
                    tokenizer.decode(generated_ids, skip_special_tokens=True)
                )
                generated_token_lengths.append(int(generated_ids.shape[0]))
    finally:
        if was_training:
            model.train()

    reward_suite = EnvironmentRewardSuite(seed=seed)
    environment_records = reward_suite.evaluate_batch(
        generated_texts,
        env_prompt=[item["prompt"] for item in prompt_data],
        task_type=[item["task_type"] for item in prompt_data],
        task_id=[item["task_id"] for item in prompt_data],
    )
    records: list[dict[str, Any]] = []
    for item, generated_text, token_length, environment_record in zip(
        prompt_data,
        generated_texts,
        generated_token_lengths,
        environment_records,
        strict=True,
    ):
        record = {
            "task_type": item["task_type"],
            "task_id": item["task_id"],
            "split": item.get("split", "unspecified"),
            "initial_score": environment_record["initial_score"],
            "reward": environment_record["environment_reward"],
            "score": environment_record["final_score"],
            "score_delta": environment_record["score_delta"],
            "valid_json": environment_record["valid_json"],
            "schema_valid": environment_record["schema_valid"],
            "used_strategy_patch": environment_record["used_patch"],
            "used_environment_patch": environment_record.get("used_environment_patch", False),
            "environment_patch_accepted": environment_record.get(
                "environment_patch_accepted", False
            ),
            "environment_learning_value": environment_record.get(
                "environment_learning_value", 0.0
            ),
            "action_origin": environment_record["action_origin"],
            "completion_tokens": token_length,
            "completion_preview": reconstruct_action_completion(
                generated_text, item["task_type"]
            )[:500],
            "grading_source": environment_record["grading_source"],
            "adapter_route": routing_policy.route_for(item["task_type"]),
        }
        if environment_record["used_patch"]:
            record["patch_accepted"] = environment_record["patch_accepted"]
            record["patch_improvement"] = environment_record["patch_improvement"]
            record["strategy_eval_source_counts"] = environment_record[
                "strategy_eval_source_counts"
            ]
        if environment_record["error"]:
            record["error"] = environment_record["error"]
        records.append(record)

    if not records:
        return {
            "mean_reward": 0.0,
            "mean_score": 0.0,
            "episodes": [],
            "generation_mode": "freeform",
            "json_action_rate": 0.0,
            "structured_action_rate": 0.0,
            "schema_valid_rate": 0.0,
            "mean_score_delta": 0.0,
            "strategy_patch_rate": 0.0,
            "patch_acceptance_rate": 0.0,
            "mean_patch_improvement": 0.0,
            "environment_patch_rate": 0.0,
            "environment_patch_acceptance_rate": 0.0,
            "mean_environment_learning_value": 0.0,
        }

    mean_reward = sum(record["reward"] for record in records) / len(records)
    mean_score = sum(record["score"] for record in records) / len(records)
    mean_score_delta = sum(record["score_delta"] for record in records) / len(records)
    task_buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        task_buckets[record["task_type"]].append(record["score"])
    task_means = {
        task_type: sum(scores) / len(scores)
        for task_type, scores in task_buckets.items()
        if scores
    }

    json_action_rate = sum(bool(record["valid_json"]) for record in records) / len(records)
    schema_valid_rate = sum(bool(record["schema_valid"]) for record in records) / len(records)
    structured_action_rate = schema_valid_rate
    patch_records = [record for record in records if record.get("used_strategy_patch")]
    accepted_patches = [record for record in patch_records if record.get("patch_accepted")]
    environment_records = [
        record for record in records if record.get("used_environment_patch")
    ]
    accepted_environments = [
        record for record in environment_records if record.get("environment_patch_accepted")
    ]
    mean_patch_improvement = (
        sum(float(record.get("patch_improvement", 0.0)) for record in patch_records) / len(patch_records)
        if patch_records
        else 0.0
    )
    grading_source_counts: dict[str, int] = defaultdict(int)
    for record in records:
        grading_source_counts[str(record.get("grading_source") or "unknown")] += 1
    
    return {
        "mean_reward": mean_reward,
        "mean_score": mean_score,
        "mean_score_delta": mean_score_delta,
        "policy_mode": policy_mode,
        "task_means": task_means,
        "json_action_rate": json_action_rate,
        "structured_action_rate": structured_action_rate,
        "schema_valid_rate": schema_valid_rate,
        "success_rate": sum(record["score"] >= 0.7 for record in records) / len(records),
        "strategy_patch_rate": len(patch_records) / len(records),
        "patch_acceptance_rate": len(accepted_patches) / len(patch_records) if patch_records else 0.0,
        "mean_patch_improvement": mean_patch_improvement,
        "environment_patch_rate": len(environment_records) / len(records),
        "environment_patch_acceptance_rate": (
            len(accepted_environments) / len(environment_records)
            if environment_records
            else 0.0
        ),
        "mean_environment_learning_value": (
            sum(float(record.get("environment_learning_value", 0.0)) for record in environment_records)
            / len(environment_records)
            if environment_records
            else 0.0
        ),
        "generation_mode": "freeform",
        "grading_source_counts": dict(grading_source_counts),
        "base_fallback_tasks": sorted(base_fallback_tasks),
        "episodes": records,
    }


def compare_paired_evaluations(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare the same held-out examples and report paired bootstrap intervals."""
    baseline_by_key = {
        (record["task_type"], record["task_id"]): record
        for record in baseline.get("episodes", [])
    }
    candidate_by_key = {
        (record["task_type"], record["task_id"]): record
        for record in candidate.get("episodes", [])
    }
    if baseline_by_key.keys() != candidate_by_key.keys():
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        raise ValueError(
            "Paired evaluation keys differ. "
            f"Missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )

    keys = sorted(baseline_by_key)
    score_deltas = [
        float(candidate_by_key[key]["score"]) - float(baseline_by_key[key]["score"])
        for key in keys
    ]
    reward_deltas = [
        float(candidate_by_key[key]["reward"]) - float(baseline_by_key[key]["reward"])
        for key in keys
    ]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _bootstrap_ci(values: list[float]) -> list[float]:
        if not values:
            return [0.0, 0.0]
        rng = random.Random(seed)
        means = []
        for _ in range(bootstrap_samples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            means.append(_mean(sample))
        means.sort()
        low_index = int(0.025 * (len(means) - 1))
        high_index = int(0.975 * (len(means) - 1))
        return [means[low_index], means[high_index]]

    per_task: dict[str, list[float]] = defaultdict(list)
    for (task_type, _), delta in zip(keys, score_deltas, strict=True):
        per_task[task_type].append(delta)

    return {
        "example_count": len(keys),
        "score_delta": _mean(score_deltas),
        "reward_delta": _mean(reward_deltas),
        "score_delta_ci95": _bootstrap_ci(score_deltas),
        "reward_delta_ci95": _bootstrap_ci(reward_deltas),
        "improved_fraction": (
            sum(delta > 1e-9 for delta in score_deltas) / len(score_deltas)
            if score_deltas
            else 0.0
        ),
        "regressed_fraction": (
            sum(delta < -1e-9 for delta in score_deltas) / len(score_deltas)
            if score_deltas
            else 0.0
        ),
        "unchanged_fraction": (
            sum(abs(delta) <= 1e-9 for delta in score_deltas) / len(score_deltas)
            if score_deltas
            else 0.0
        ),
        "per_task_score_delta": {
            task_type: _mean(deltas) for task_type, deltas in sorted(per_task.items())
        },
    }


def select_and_gate_candidate(
    baseline: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Select the strongest safe stage and refuse capability regressions."""
    if not candidates:
        raise ValueError("At least one candidate stage is required")

    def _rank(name: str) -> tuple[float, float, float]:
        return (
            float(candidates[name].get("mean_score", 0.0)),
            float(candidates[name].get("mean_reward", 0.0)),
            float(candidates[name].get("schema_valid_rate", 0.0)),
        )

    stage_decisions: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        comparison = compare_paired_evaluations(baseline, candidate)
        reasons: list[str] = []
        if comparison["score_delta"] <= 0.0:
            reasons.append("held-out mean score did not improve")
        if comparison["reward_delta"] < 0.0:
            reasons.append("held-out mean environment reward regressed")
        if float(candidate.get("schema_valid_rate", 0.0)) + 0.05 < float(
            baseline.get("schema_valid_rate", 0.0)
        ):
            reasons.append(
                "strict action-schema validity regressed by more than 5 percentage points"
            )
        if comparison["regressed_fraction"] > 0.25:
            reasons.append("more than 25% of held-out examples regressed")
        if comparison["improved_fraction"] <= comparison["regressed_fraction"]:
            reasons.append("held-out improvements did not outnumber regressions")
        for task_type, delta in comparison["per_task_score_delta"].items():
            if delta < -0.02:
                reasons.append(
                    f"task family {task_type} regressed by {abs(delta):.4f} (> 0.0200 tolerance)"
                )
        stage_decisions[name] = {
            "passed": not reasons,
            "reasons": reasons,
            "comparison": comparison,
        }

    eligible = [name for name, decision in stage_decisions.items() if decision["passed"]]
    selected_stage = max(eligible or list(candidates), key=_rank)
    selected = candidates[selected_stage]
    selected_decision = stage_decisions[selected_stage]
    comparison = selected_decision["comparison"]
    reasons = selected_decision["reasons"]

    promoted = bool(eligible)
    model_recursive_evidence = bool(
        promoted
        and float(selected.get("strategy_patch_rate", 0.0)) > 0.0
        and float(selected.get("patch_acceptance_rate", 0.0)) > 0.0
        and float(selected.get("mean_patch_improvement", 0.0)) > 0.0
    )
    environment_recursive_evidence = bool(
        promoted
        and float(selected.get("environment_patch_rate", 0.0)) > 0.0
        and float(selected.get("environment_patch_acceptance_rate", 0.0)) > 0.0
        and float(selected.get("mean_environment_learning_value", 0.0)) > 0.0
    )
    coevolution_evidence = model_recursive_evidence and environment_recursive_evidence
    return {
        "selected_stage": selected_stage,
        "promoted": promoted,
        "reasons": reasons,
        "comparison": comparison,
        "stage_decisions": stage_decisions,
        "recursive_evidence": model_recursive_evidence,
        "model_recursive_evidence": model_recursive_evidence,
        "environment_recursive_evidence": environment_recursive_evidence,
        "coevolution_evidence": coevolution_evidence,
        "evidence_label": (
            "verified_coevolution"
            if coevolution_evidence
            else "recursive_model_improvement"
            if model_recursive_evidence
            else "recursive_environment_improvement"
            if environment_recursive_evidence
            else "heldout_capability_improvement"
            if promoted
            else "rejected_regression"
        ),
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
    capability_rewards = [
        item.get("rewards/capability_delta/mean", None)
        for item in grpo_logs
        if "reward" in item
    ]
    structure_rewards = [
        item.get("rewards/strict_structure/mean", None)
        for item in grpo_logs
        if "reward" in item
    ]

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
        if any(v is not None for v in capability_rewards):
            ax.plot(
                reward_steps,
                [v or 0 for v in capability_rewards],
                marker="s",
                color="#0f766e",
                linewidth=1.5,
                alpha=0.8,
                label="Capability delta reward",
            )
        if any(v is not None for v in structure_rewards):
            ax.plot(
                reward_steps,
                [v or 0 for v in structure_rewards],
                marker="^",
                color="#ea580c",
                linewidth=1.5,
                alpha=0.7,
                label="Strict schema",
            )
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
    behavior_labels = [
        "Schema\nvalid",
        "Strategy\nscore",
        "Model patch\naccepted",
        "Env patch\naccepted",
        "Env learning\nvalue",
    ]
    baseline_recursive = [
        baseline_metrics.get("structured_action_rate", 0.0),
        baseline_metrics.get("task_means", {}).get("strategy_optimization", 0.0),
        baseline_metrics.get("patch_acceptance_rate", 0.0),
        baseline_metrics.get("environment_patch_acceptance_rate", 0.0),
        baseline_metrics.get("mean_environment_learning_value", 0.0),
    ]
    trained_recursive = [
        trained_metrics.get("structured_action_rate", 0.0),
        trained_metrics.get("task_means", {}).get("strategy_optimization", 0.0),
        trained_metrics.get("patch_acceptance_rate", 0.0),
        trained_metrics.get("environment_patch_acceptance_rate", 0.0),
        trained_metrics.get("mean_environment_learning_value", 0.0),
    ]
    x = range(len(behavior_labels))
    ax.bar([i - 0.18 for i in x], baseline_recursive, width=0.35, label="Baseline", color="#94a3b8")
    ax.bar([i + 0.18 for i in x], trained_recursive, width=0.35, label="Trained", color="#0f766e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(behavior_labels, fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Recursive Policy Behavior")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
