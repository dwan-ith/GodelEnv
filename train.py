"""
Godel Env -- TRL/GRPO Training Script

Trains an LLM agent to improve solutions inside the GodelEnv RL environment
using Group Relative Policy Optimization (GRPO) via Hugging Face TRL + Unsloth.

Training runs against a REMOTE GodelEnv deployed on Hugging Face Spaces.

Usage:
    # Dry run (validates setup without GPU)
    python train.py --dry-run

    # Full training against remote HF Space
    python train.py --space-url https://litterarum-godelenv.hf.space

    # With custom model
    python train.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --epochs 3

    # Resume from checkpoint
    python train.py --resume-from checkpoints/checkpoint-500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=False)

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("godel_train")

# ---------------------------------------------------------------------------
# Lazy imports (only load heavy ML libs when actually training)
# ---------------------------------------------------------------------------


def _check_dependencies():
    """Verify training dependencies are installed."""
    missing = []
    for pkg in ["trl", "transformers", "datasets", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing training dependencies: {', '.join(missing)}")
        print("Install them with: pip install 'godel_engine[train]'")
        print("  or: pip install trl transformers datasets torch")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_prompt_dataset(
    space_url: str,
    num_prompts: int = 50,
    tasks: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Build a dataset of prompts by resetting the remote GodelEnv repeatedly.
    Each prompt contains the task description + initial solution.
    """
    from godel_engine.client import GodelEngineEnv
    from godel_engine.rollout import build_prompt

    all_tasks = tasks or [
        "factual_qa", "alignment_qa", "code_improvement",
        "python_optimized", "reasoning", "adr_writing",
        "strategy_optimization",
    ]

    dataset = []
    loop = asyncio.new_event_loop()

    async def _collect():
        async with GodelEngineEnv(base_url=space_url) as client:
            for i in range(num_prompts):
                task_type = all_tasks[i % len(all_tasks)]
                try:
                    result = await client.reset(task_type=task_type)
                    obs = result.observation
                    prompt_text = build_prompt(
                        task_prompt=obs.task_prompt,
                        current_solution=obs.current_solution,
                        rubric_feedback=obs.feedback_summary,
                    )
                    dataset.append({
                        "prompt": prompt_text,
                        "task_type": task_type,
                        "task_id": obs.task_id,
                        "initial_score": str(obs.total_score),
                    })
                    logger.info(f"Collected prompt {i+1}/{num_prompts} ({task_type})")
                except Exception as e:
                    logger.warning(f"Failed to collect prompt {i+1}: {e}")

    loop.run_until_complete(_collect())
    loop.close()
    logger.info(f"Built dataset with {len(dataset)} prompts")
    return dataset


# ---------------------------------------------------------------------------
# Rollout function for GRPOTrainer
# ---------------------------------------------------------------------------


def make_grpo_rollout(space_url: str, max_env_steps: int = 3):
    """
    Create a rollout function for TRL's GRPOTrainer.

    The rollout function:
      1. Takes prompts from the trainer
      2. Generates completions using the model
      3. Steps through the remote GodelEnv
      4. Returns rewards per channel
    """
    from godel_engine.client import GodelEngineEnv
    from godel_engine.rollout import parse_completion_to_action, extract_reward_channels

    def rollout_func(prompts: List[str], trainer) -> Dict[str, Any]:
        """Custom rollout that interacts with remote GodelEnv."""
        tokenizer = trainer.processing_class

        # Generate completions using the model
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode completions (strip the prompt portion)
        prompt_lengths = inputs["input_ids"].shape[1]
        completion_ids = outputs[:, prompt_lengths:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Step through the environment for each completion
        all_rewards = {
            "task_score_delta": [],
            "format_compliance": [],
            "anti_hack_penalty": [],
            "env_score": [],
        }

        loop = asyncio.new_event_loop()

        async def _step_all():
            async with GodelEngineEnv(base_url=space_url) as client:
                for i, completion in enumerate(completions):
                    try:
                        # Reset env for this prompt
                        await client.reset()
                        # Parse completion into action
                        action = parse_completion_to_action(completion)
                        # Step
                        result = await client.step(action)
                        channels = extract_reward_channels(result)
                        for key in all_rewards:
                            all_rewards[key].append(channels.get(key, 0.0))
                    except Exception as e:
                        logger.warning(f"Rollout step {i} failed: {e}")
                        for key in all_rewards:
                            all_rewards[key].append(0.0)

        loop.run_until_complete(_step_all())
        loop.close()

        # Compute log probabilities
        import torch
        with torch.no_grad():
            model_outputs = trainer.model(
                input_ids=outputs,
                attention_mask=torch.ones_like(outputs),
            )
            logits = model_outputs.logits[:, prompt_lengths - 1:-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)

        return {
            "prompt_ids": inputs["input_ids"].tolist(),
            "completion_ids": completion_ids.tolist(),
            "logprobs": token_log_probs.tolist(),
            **all_rewards,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args):
    """Execute the full training pipeline."""
    _check_dependencies()

    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import GRPOTrainer, GRPOConfig
    from godel_engine.rollout import (
        task_reward_func,
        format_reward_func,
        guard_reward_func,
        score_reward_func,
    )

    logger.info("=" * 60)
    logger.info("  GODEL ENV -- GRPO TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Space URL:  {args.space_url}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Prompts:    {args.num_prompts}")
    logger.info("=" * 60)

    # Load model with Unsloth for efficiency (if available)
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )
        logger.info("Loaded model with Unsloth (4-bit + LoRA)")
    except ImportError:
        logger.warning("Unsloth not available. Using standard HF loading.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompt dataset from remote env
    logger.info("Collecting prompts from remote GodelEnv...")
    prompt_data = build_prompt_dataset(
        space_url=args.space_url,
        num_prompts=args.num_prompts,
        tasks=args.tasks.split(",") if args.tasks else None,
    )

    if not prompt_data:
        logger.error("No prompts collected. Check Space URL and connectivity.")
        sys.exit(1)

    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in prompt_data])

    # Configure GRPO training
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=50,
        max_completion_length=1024,
        num_generations=args.num_generations,
        report_to="wandb" if args.wandb else "none",
    )

    # Build rollout function
    rollout_fn = make_grpo_rollout(
        space_url=args.space_url,
        max_env_steps=args.max_env_steps,
    )

    # Initialize trainer with multiple independent reward functions
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        rollout_func=rollout_fn,
        reward_funcs=[
            task_reward_func,
            format_reward_func,
            guard_reward_func,
            score_reward_func,
        ],
    )

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save the trained model
    logger.info(f"Saving model to {args.output_dir}/final")
    try:
        from unsloth import FastLanguageModel
        # Proper LoRA merge for Unsloth models
        model.save_pretrained_merged(
            f"{args.output_dir}/final",
            tokenizer,
            save_method="merged_16bit",
        )
        logger.info("Model saved with Unsloth merged-16bit method")
    except (ImportError, AttributeError):
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        logger.info("Model saved with standard HF method")

    logger.info("Training complete!")


def dry_run(args):
    """Validate setup without requiring GPU."""
    logger.info("=" * 60)
    logger.info("  DRY RUN -- Validating setup")
    logger.info("=" * 60)

    # Check dependencies
    _check_dependencies()
    logger.info("[OK] All training dependencies installed")

    # Check env connectivity
    from godel_engine.client import GodelEngineEnv
    loop = asyncio.new_event_loop()

    async def _test():
        async with GodelEngineEnv(base_url=args.space_url) as client:
            result = await client.reset(task_type="factual_qa")
            logger.info(f"[OK] Remote env reset successful")
            logger.info(f"     Task: {result.observation.task_type}")
            logger.info(f"     Score: {result.observation.total_score:.3f}")
            logger.info(f"     Prompt: {result.observation.task_prompt[:80]}...")
            return True

    try:
        loop.run_until_complete(_test())
    except Exception as e:
        logger.error(f"[FAIL] Cannot connect to {args.space_url}: {e}")
        sys.exit(1)
    finally:
        loop.close()

    # Check rollout parsing
    from godel_engine.rollout import parse_completion_to_action
    test_completion = '{"solution": "test answer", "edit_type": "rewrite", "strategy_note": "test"}'
    action = parse_completion_to_action(test_completion)
    logger.info(f"[OK] Action parsing works: {action.edit_type.value}")

    # Check reward functions
    from godel_engine.rollout import ALL_REWARD_FUNCS
    logger.info(f"[OK] {len(ALL_REWARD_FUNCS)} reward functions registered")

    logger.info("=" * 60)
    logger.info("  All checks passed. Ready to train!")
    logger.info("  Run without --dry-run to start training.")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LLM agent with GRPO on GodelEnv"
    )
    parser.add_argument(
        "--space-url", type=str,
        default=os.getenv("GODEL_SPACE_URL", "https://litterarum-godelenv.hf.space"),
        help="URL of the remote GodelEnv HF Space",
    )
    parser.add_argument(
        "--model", type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Model to train (HF model ID or local path)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO")
    parser.add_argument("--max-env-steps", type=int, default=3,
                        help="Max environment steps per rollout")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task types to train on")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without training")

    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        train(args)
