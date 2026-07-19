---
title: GodelEnv
emoji: "🔁"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# GodelEnv: A Self-Improving RL Environment

GodelEnv is a RL Environment for environment-agent coevolution. Diverging from traditional static benchmarks, GodelEnv operationalizes a dual-action reinforcement learning (RL) framework where the learning agent simultaneously mutates its reasoning policy and the curriculum it is evaluated on. 

The environment institutes two coupled, empirically governed improvement loops:

1. **Policy Mutation (`StrategyPatch`)**: The model proposes modifications to its own reasoning traces. Parent and child strategies are iteratively replayed against a held-out evaluation substrate, and the mutation is only accepted if it demonstrates a statistically significant multi-objective utility improvement without catastrophic regression.
2. **Curriculum Evolution (`EnvironmentPatch`)**: The model constructs novel challenges by combining and mutating immutable, verified base tasks. The mutation is admitted to the active challenge pool only if it exhibits measurable regret (i.e., solvable by the environment's baseline solver but challenging for the current agent frontier).

Both loops are intrinsically falsifiable. Generated completions cannot manipulate graders, evaluation references, terminal reward allocation functions, or Governor threshold bounds. 

## Documentation and Infrastructure

- **Live Inference / Hugging Face Space**: [https://huggingface.co/spaces/litterarum/GodelEnv](https://huggingface.co/spaces/litterarum/GodelEnv)
- **Source Repository**: [https://github.com/dwan-ith/GodelEnv](https://github.com/dwan-ith/GodelEnv)
- **Trainable Colab Implementation**: [train_colab.ipynb](train_colab.ipynb)
- **Methodology Write-up**: [blog.md](blog.md)
- **Promoted Policy Routing Manifest**: [artifacts/training_run/routing.json](artifacts/training_run/routing.json)

**Training Evidence Logs:**
- **Training Metrics**: [artifacts/training_run/metrics.json](artifacts/training_run/metrics.json)
- **Loss Progression**: [artifacts/training_run/loss_curve.png](artifacts/training_run/loss_curve.png)
- **GRPO Reward Trajectory**: [artifacts/training_run/reward_curve.png](artifacts/training_run/reward_curve.png)
- **Held-Out Distribution Shift (Before/After)**: [artifacts/training_run/before_after.png](artifacts/training_run/before_after.png)
- **Deterministic Coevolution Mechanism Validation**: [artifacts/coevolution_smoke_v2/metrics.json](artifacts/coevolution_smoke_v2/metrics.json)
- **Coevolution Acceptance Curve**: [artifacts/coevolution_smoke_v2/coevolution_curve.png](artifacts/coevolution_smoke_v2/coevolution_curve.png)

## Theoretical Architecture

The architecture mathematically couples principles from [POET](https://arxiv.org/abs/1901.01753) (Paired Open-Ended Trailblazer), which coordinates generative environments with agent optimization, and [PAIRED](https://arxiv.org/abs/2012.02096), which bounds adversarial environments via calculable regret. 

```text
  Observation: Current Strategy + Downstream Scores + Failure Vectors 
                               |
                               v
                     LLM Generates Action
                 /                            \
                v                              v
       StrategyPatch                    EnvironmentPatch
  (Revised reasoning policy)     (Bounded curriculum mutation)
                |                              |
                v                              v
       STRATEGY GOVERNOR              ENVIRONMENT GOVERNOR
  Examines utility delta          Maintains challenge solvability
  against held-out targets        bounds and computes Regret
                |                              |
                v                              v
  StrategyRegistry (Elo tracking)   ChallengePool (Priority replay)
```

### Action Space Constraints

#### StrategyPatch Verification
A valid strategy episode must yield a complete revised strategy, a structured difference description, an explicit hypothesis, and targeted weaknesses. The Strategy Governor executes deterministic evaluations across a rotating subset of tasks: Factual QA, Alignment, Reasoning, Code Improvement, Python Optimization, and ADR Writing. Acceptance strictly mandates cross-domain variance stability, no systematic regressions, and zero evaluator leakage.

#### EnvironmentPatch Verification
The agent specifies bounded curriculum mutations via JSON structures mapping to functional operations (`deepen`, `contrast`, `transfer`) against immutable verified instances. The environment retains total control over the reference mappings. Admissions require a provable delta in current-policy performance and positive alignment to the requested capability frontier.

## Reward Formulations and Anti-Hacking Guardrails

To train the policy utilizing TRPO/GRPO (Group Relative Policy Optimization), GodelEnv projects six independent reward channels:

1. `capability_delta`: Differential improvement in zero-shot task scoring.
2. `absolute_quality`: Terminal absolute utility of the generated artifact.
3. `strict_structure`: Enforces JSON syntactical integrity (malformed outputs instantly receive 0.0).
4. `safety`: Penalizes guardrail violations, impermissible code, leakage, or attempts to manipulate runtime variables.
5. `recursive_patch`: Reward allocation contingent upon the Strategy Governor accepting a parent-child logic displacement.
6. `environment_patch`: Reward allocation contingent upon the Environment Governor admitting a valid frontier challenge.

Actions process through isolated deterministic solvers. Security domains, `ast`-parsed syntax checking, and environment timeouts prevent reward hacking. 

## Hybrid Dual-Action Runtime Strategy

GodelEnv implements fail-safe deterministic solvers for local RLVR scoring and programmatic validation (`GODEL_GRADING_MODE="deterministic"` or `GODEL_STRATEGY_EVAL_MODE="deterministic"`). The transition to LLM-as-a-judge is seamlessly supported up to the API routing layer; however, deterministic evaluation prevents conflating model evaluation bias with authentic capability growth during training.

```bash
$env:GODEL_AGENT_MODE="deterministic"
$env:GODEL_GRADING_MODE="deterministic"
$env:GODEL_ALLOW_DETERMINISTIC_FALLBACK="1"
python self_improve.py --iterations 6 --max-patch-attempts 2 \
  --registry-path artifacts/coevolution_smoke_v2/strategy_registry.json \
  --challenge-archive-path artifacts/coevolution_smoke_v2/challenge_archive.json \
  --metrics-path artifacts/coevolution_smoke_v2/metrics.json \
  --plot-path artifacts/coevolution_smoke_v2/coevolution_curve.png
```

## Empirical Evidence

The training baseline utilized `Qwen/Qwen2.5-0.5B-Instruct` on low-parameter compute (80 SFT steps, 12 repair steps, 6 GRPO steps) across 32 structured prompts, evaluated against a disjoint set of 8 independent held-out examples. 

### Quantitative Baseline Deltas

| Held-out Metric | Untrained Baseline | Routed GRPO Policy | Capability Delta |
| :--- | :---: | :---: | :---: |
| **Mean Task Score** | 0.4356 | 0.4510 | +0.0155 |
| **Mean Environment Reward** | -0.0589 | 0.0962 | +0.1551 |
| **Schema Integrity Rate** | 50.0% | 87.5% | +37.5 pp |
| **Strategy Patch Acceptance** | 0.0% | 50.0% | +50.0 pp |
| **Environment Patch Acceptance**| 0.0% | 50.0% | +50.0 pp |

### Critical Analysis and Statistical Limits

**Validity of Data**: 
The GRPO routing achieved a **Positive Environment Reward** delta bounded by $95\%$ Confidence Intervals of $[+0.0095, +0.3217]$. This certifies, at a $p < .05$ threshold, that the model statistically optimized for the objective function. Furthermore, the model learned to simultaneously emit viable `StrategyPatch` and `EnvironmentPatch` representations, achieving a unified `verified_coevolution` label that required strict passage through both evaluation Governors.

**Caveats & Areas of Friction**:
1. **Sample Constraint**: The Task Score delta ($+0.0155$) entails a $95\%$ Confidence Interval that encompasses zero (`CI: [-0.0185, +0.0617]`). Due to computational limits confining the held-out sample batch to $N=8$, the raw capability estimate remains statistically indeterminate without scaling the evaluation batch to larger sample sizes.
2. **Combinatorial Saturation**: Curriculum evolution operates over a discrete foundation (e.g., source targets `qa01`-`qa08`). Long-term continuous pretraining (over hundreds of thousands of gradient updates) may result in diminishing marginal utility as the permutations of source IDs saturate.

## Run and Reproduce

Install dependencies via `uv` and initialize the stack:

```bash
uv sync --extra dev --extra train
openenv validate
pytest -q --basetemp="tmp_pytest"
python -m compileall godel_engine server train.py train_colab.py
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

*Note: The FastAPI WebSockets dashboard handles multi-step telemetry logging natively over `/ws` and exposes provider diagnostics externally at `/demo/provider-status`.*

See the declarative OpenEnv compliance matrix at [openenv.yaml](openenv.yaml).
