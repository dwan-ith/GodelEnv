---
title: GodelEnv
emoji: 🔁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# GodelEnv: A Verifiable Environment for Recursive Self-Improvement

GodelEnv is a reinforcement learning environment built on the `openenv-core` framework. It is designed to move beyond static, single-turn task optimization. Instead of merely rewarding an agent for producing a correct final answer, GodelEnv provides the infrastructure for **recursive skill amplification**: an agent can propose mutations to its own reasoning strategy, test those mutations against hidden downstream tasks, and retain them only if they genuinely improve multi-objective utility.

## The Problem

Most LLM training environments focus on evaluating outputs against a fixed rubric (e.g., did the code pass the test? is the QA factually correct?). While effective for skill acquisition, this approach does not teach an agent *how to learn* or *how to adapt its reasoning*.

GodelEnv introduces a meta-loop. In a `strategy_optimization` episode, the agent sees its current reasoning strategy, recent failure cases, and downstream scores. It can then output a `StrategyPatch`. 

Instead of blindly accepting the mutation, the environment plays the role of a rigorous experimental sandbox:
1. It builds a child strategy from the patch.
2. It evaluates the parent vs. the child strategy on a bundle of held-out, episodic downstream tasks.
3. A Governor applies regression gates and anti-hacking guards.
4. The patch is accepted only if it demonstrates verifiable, generalized improvement.

This shifts the RL objective from "answering a question" to "improving the policy that answers questions."

## Architecture

GodelEnv behaves as a fully compliant OpenEnv service, making it compatible with existing agentic RL pipelines.

- **OpenEnv Interface**: Conforms strictly to `openenv-core>=0.2.3` standards with standard `/reset`, `/step`, and `/state` API surfaces.
- **Hybrid Runtime**: Supports a dual-execution path. It uses an LLM-first behavior (`auto`) for rigorous grading, agent actions, and strategy evaluation, but gracefully fails over to deterministic grading if provider credentials are stale or rate-limited.
- **Multi-Channel Reward**: Discards the single scalar reward in favor of a detailed, multi-axis vector including `task_score_delta`, `format_compliance`, `generalization_score`, `robustness_score`, and `anti_hack_penalty`.
- **Guardrails**: Includes robust checks for empty responses, repetition, forbidden code patterns, canary leakage, and strategy length limits.

### Supported Providers

The environment supports rapid iteration across major inference providers:

```bash
# OpenAI Configuration
set OPENAI_API_KEY=...
set OPENAI_MODEL_NAME=gpt-4o-mini

# Custom/vLLM/Ollama Configuration
set API_KEY=...
set API_BASE_URL=https://your-openai-compatible-provider/v1
set CUSTOM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Hugging Face Router Configuration
set HF_TOKEN=...
set HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
set HF_API_BASE_URL=https://router.huggingface.co/v1
```

Common runtime flags to control the hybrid execution:
```bash
set GODEL_GRADING_MODE=auto
set GODEL_STRATEGY_EVAL_MODE=auto
set GODEL_PROVIDER_ORDER=openai,custom,huggingface
```

## Verifiable Training Pipeline

To prove the environment mechanics are learnable, GodelEnv ships with a reproducible training pipeline. The pipeline uses a compact action space so a tiny, local CPU policy can be trained and verified without massive compute.

1. **Prompt Collection**: Samples initial states from the live environment.
2. **Warm-Start**: Generates heuristic traces to teach the model the base `Action` or `StrategyPatch` schema.
3. **SFT**: Supervised fine-tuning of a tiny local policy.
4. **GRPO**: Group Relative Policy Optimization against the live environment to maximize the multi-channel reward.

### Training Evidence

The committed training evidence (available in `artifacts/training_run`) demonstrates that the environment provides a clear, learnable gradient. The results below compare a random baseline against the trained policy over 16 prompts across factual QA, alignment QA, reasoning, and strategy optimization tasks.

| Metric | Baseline | Trained | Delta |
| --- | ---: | ---: | ---: |
| Mean reward | 0.4090 | 0.5224 | **+0.1134** |
| Mean score | 0.7361 | 0.8384 | **+0.1023** |

#### Loss Curve (SFT)
![SFT loss curve](artifacts/training_run/loss_curve.png)

#### Reward Curve (GRPO)
![GRPO reward curve](artifacts/training_run/reward_curve.png)

#### Comparison
![Before and after training](artifacts/training_run/before_after.png)

This shows concrete learning on both reward and downstream score. The current miniature policy converges to a conservative "direct-answer" strategy. The environment path is fully verified and ready for scaling with larger models that can explore richer recursive behaviors.

## Real-Time Dashboard

The environment includes a high-fidelity dashboard to monitor the recursive loop visually:

- **Engine Status**: Real-time tracking of score progression, step counts, and delta.
- **Strategy Stats**: Monitors the agent's Strategy ELO, Generation lineage, and remaining computation budget.
- **Reasoning Strategy**: Displays the specific reasoning policy currently driving the agent's behavior.
- **Recent Failures**: A live feedback loop showing the "challengers" (failed tasks) that guide the next strategy mutation.
- **Telemetry**: Verbose streaming of grading sources and Governor acceptance/rejection decisions.

## Quick Start & Validation

### Run the Environment Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Validate the Engine
Ensure all core mechanics, grading paths, and OpenEnv specifications are functioning:
```bash
openenv validate
pytest -q
python -m compileall godel_engine server train.py train_colab.py demo.py
```

### Run the Training Pipeline
Train the local proof-of-concept policy and export training plots:
```bash
python train.py
```
Or use the pre-configured notebook: `train_colab.ipynb`

## Project Links

- **Hosted Demo**: [HF Space](https://huggingface.co/spaces/litterarum/GodelEnv)
- **Source Code**: [GitHub](https://github.com/dwan-ith/GodelEnv)
- **Blog**: [blog_draft.md](blog_draft.md)
