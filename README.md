---
title: GodelEnv
emoji: "🚞"
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
---
# Godel Env

> A self-improving reinforcement learning environment built on [OpenEnv](https://openenv.dev).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.dev)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An LLM agent iteratively improves solutions across professional real-world domains. Each step is
graded by a multi-axis LLM-as-a-judge rubric grader, producing partial progress signals (not
sparse binary rewards). The hardest task ("Godel tier") is fully recursive: the agent must improve
a reasoning template that is then empirically tested on a downstream challenge.

---

## Architecture

```
godel_engine/               <- Core RL package (no web server needed)
|   environment.py          <- GodelEnvironment: reset() / step() / state()
|   agent.py                <- AutoAgent: LLM-based action policy
|   evolution.py            <- DarwinPool + HuxleyTracker (strategy evolution)
|   models.py               <- Typed Pydantic models (GodelAction, RewardBreakdown...)
|   guards.py               <- Anti-reward-hacking guards (5 independent checks)
|   curriculum.py           <- Automatic difficulty progression controller
|   rollout.py              <- TRL/GRPO rollout integration + reward functions
|   client.py               <- HTTP client for remote env access
|   graders/
|       agent_grader.py     <- LLM-as-a-judge grader (OpenAI client)
|   tasks/
|       base.py             <- BaseTask abstract class
|       factual_qa.py       <- Easy: factual explanation
|       alignment_qa.py     <- Easy: AI safety concepts
|       code_improvement.py <- Medium: Python correctness (deterministic)
|       python_optimized.py <- Medium: code performance + docs
|       reasoning.py        <- Medium: multi-step logic
|       adr_writing.py      <- Hard: Architecture Decision Records
|       strategy_optimization.py <- Godel: recursive self-improvement
server/                     <- FastAPI dashboard wrapper
train.py                    <- TRL/GRPO training script (Unsloth + LoRA)
demo.py                     <- Before/after demonstration
inference.py                <- Hackathon inference script ([START]/[STEP]/[END] format)
baseline.py                 <- Standalone RL evaluation script
```

---

## Tasks

| Task | Difficulty | Grader | Rubrics |
|---|---|---|---|
| `factual_qa` | Easy | LLM judge | coverage, detail, structure |
| `alignment_qa` | Easy | LLM judge | technical_accuracy, clarity, completeness |
| `code_improvement` | Medium | Deterministic (AST + exec) | syntax, tests, documentation |
| `python_optimized` | Medium | LLM judge | correctness, efficiency, documentation |
| `reasoning` | Medium | LLM judge | logical_validity, completeness, clarity |
| `adr_writing` | Hard | LLM judge | structure, completeness, trade_off_analysis |
| `strategy_optimization` | Godel | Two-phase (structural + empirical) | self_verification, structural_rigor, recursive_potential, downstream_quality |

---

## Reward Channels

GodelEnv uses **multiple independent reward functions** instead of a single scalar.
Each channel can be used separately by TRL's GRPOTrainer for multi-objective optimization.

| Channel | Range | Description |
|---|---|---|
| `task_score_delta` | [-1, 1] | Score improvement from rubric grading |
| `format_compliance` | [0, 0.02] | Bonus for following expected output format |
| `step_cost` | -0.005 | Fixed per-step penalty (encourages efficiency) |
| `anti_hack_penalty` | [-1, 0] | Penalty from anti-reward-hacking guards |
| `process_reward` | [0, 0.05] | Step-level reasoning quality bonus |
| `total` | sum | Sum of all channels |

---

## Anti-Reward-Hacking Guards

Five independent guards run on every `step()` call:

| Guard | Penalty | What It Catches |
|---|---|---|
| Empty solution | -0.5 | Whitespace-only or empty submissions |
| Length guard | -0.3 | Solutions >10x or <0.1x initial length |
| Repetition guard | -0.4 | Copy-paste spam (trigram repetition >40%) |
| Forbidden patterns | -0.5 | `import os`, `exec()`, `globals()` in code tasks |
| Regression guard | -0.2 | Score drops >0.3 in one step |

Severe violations (penalty <= -0.8) terminate the episode immediately.

---

## Curriculum Learning

The environment automatically adjusts task difficulty based on agent performance:

```
easy -> medium -> hard -> godel
```

- Escalation: success_rate > 0.6 over last 10 episodes
- De-escalation: success_rate < 0.2 over last 10 episodes
- Manual override: `env.reset(difficulty="hard")`

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/dwan-ith/GodelEnv.git
cd GodelEnv
pip install -e ".[dev]"
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and set:
# HF_TOKEN=hf_...
# API_BASE_URL=https://router.huggingface.co/v1
# MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

### 3. Run Demo

```bash
python demo.py
```

### 4. Run Baseline

```bash
python baseline.py --tasks factual_qa code_improvement reasoning --seed 42
```

### 5. Run Dashboard

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
# Open: http://localhost:7860/dashboard/
```

---

## Training with TRL/GRPO

GodelEnv integrates with Hugging Face TRL for reinforcement learning training
using Group Relative Policy Optimization (GRPO) + Unsloth for efficiency.

### Install Training Dependencies

```bash
pip install -e ".[train]"
```

### Validate Setup (Dry Run)

```bash
python train.py --dry-run --space-url https://litterarum-godelenv.hf.space
```

### Train

```bash
# Full training against remote HF Space
python train.py \
  --space-url https://litterarum-godelenv.hf.space \
  --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
  --epochs 3 \
  --batch-size 2 \
  --num-prompts 50

# Resume from checkpoint
python train.py --resume-from checkpoints/checkpoint-500
```

The training script:
1. Loads the model with Unsloth (4-bit quantization + LoRA)
2. Collects prompts from the remote GodelEnv
3. Runs GRPO with 4 independent reward functions
4. Saves the trained model with proper LoRA merge

---

## Use as a Python Library

```python
import asyncio
from godel_engine import GodelEnvironment, AutoAgent, GodelAction

async def main():
    env = GodelEnvironment(seed=42)
    agent = AutoAgent()

    result = await env.reset(task_type="code_improvement")
    print(f"Task: {result.observation.task_prompt}")
    print(f"Baseline score: {result.observation.total_score:.3f}")
    print(f"Difficulty: {result.info['difficulty']}")

    while not result.terminated and not result.truncated:
        action = await agent.act(
            task_prompt=result.observation.task_prompt,
            current_solution=result.observation.current_solution,
            rubrics=result.observation.rubric_scores.scores,
            task_type=result.observation.task_type,
        )
        result = await env.step(action)
        print(f"Step {result.observation.step}: "
              f"score={result.observation.total_score:.3f} "
              f"reward={result.reward:+.4f} "
              f"reason={result.info['reason']}")

        # Inspect reward breakdown
        rb = result.reward_breakdown
        print(f"  task_delta={rb.task_score_delta:+.3f} "
              f"format={rb.format_compliance:+.3f} "
              f"guard={rb.anti_hack_penalty:+.3f}")

    # Check curriculum state
    print(f"Curriculum: {env.curriculum.get_stats()}")

asyncio.run(main())
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes (HF) | Hugging Face token used as API key |
| `API_BASE_URL` | Yes (HF) | LLM endpoint, e.g. `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Yes (HF) | Model identifier, e.g. `Qwen/Qwen2.5-7B-Instruct` |
| `GODEL_SPACE_URL` | For training | Remote GodelEnv Space URL |

---

## Docker

```bash
docker build -t godel-env .
docker run -p 7860:7860 -e HF_TOKEN=hf_... -e API_BASE_URL=... -e MODEL_NAME=... godel-env
```

---

## Evolutionary Layers

- **Darwin Pool**: Tournament-based strategy selection. Strategies that produce higher scores survive and spawn children.
- **Huxley Tracker**: Measures Clade-Metaproductivity (CMP) -- strategies that produce better descendants are prioritized, not just strategies with high personal scores.

---

## Reproducible Baseline

```bash
python baseline.py --episodes 7 --seed 42 --output baseline_results.json
```

| Task | Difficulty | Avg Score | Avg Delta |
|---|---|---|---|
| factual_qa | Easy | ~0.72 | +0.67 |
| alignment_qa | Easy | ~0.68 | +0.63 |
| code_improvement | Medium | ~0.80 | +0.60 |
| python_optimized | Medium | ~0.65 | +0.55 |
| reasoning | Medium | ~0.63 | +0.53 |
| adr_writing | Hard | ~0.58 | +0.48 |
| strategy_optimization | Godel | ~0.71 | +0.63 |

*(Scores vary with model and temperature.)*
