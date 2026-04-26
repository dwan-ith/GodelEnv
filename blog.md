# [Mini-Blog] A Training Environment Where "Better" Has to Mean Something

When you train an LLM, you are not just teaching it to answer questions. You are shaping a latent *procedure* for how it approaches questions like what steps it runs through, which failure modes it slides past, how it balances depth against speed. Most RL environments treat that procedure as a black box. You observe a reply, assign a reward, and nudge the weights. The policy implicitly shifts, but you never directly negotiate with it.

GodelEnv makes that procedure a first-class object of training. During an episode, the agent has two available actions:
- improve the current answer directly, or 
- submit a `StrategyPatch`, a structured proposal that names what it would change about its own reasoning policy, why it expects the change to help, and which documented weaknesses it targets. 

The environment's response to a patch is not to accept it optimistically but rather it runs a comparative evaluation between the current policy and the proposed one on a held-out task bundle, then applies a multi-objective acceptance criterion. The patch survives only if it demonstrates generalized improvement without regressions across the other task domains.

That bar is hard to clear, and intentionally so. The naive failure mode of any self-modification scheme is that the agent learns to generate plausible-sounding improvements without actually producing them, with more elaborate language, longer reasoning chains, confident presentation, etc. GodelEnv counters this by using: 
- The Governor that evaluates patches checks for cross-domain regression (a patch that helps reasoning but hurts factual QA fails outright), canary-based data leakage, and variance inflation. 
- Reward is decomposed into a vector rather than a scalar: `task_score_delta`, `format_compliance`, `patch_quality`, `generalization_score`, and `anti_hack_penalty`. 

When alignment QA and reasoning diverge in opposite directions, you see it in the logs rather than watching it disappear into an average.

### Architecture & Training

The environment is OpenEnv-compliant — standard `reset`, `step`, and `state` over HTTP — so existing TRL-based training pipelines connect without custom instrumentation. Evaluation runs against a mix of factual QA, alignment-style tasks, multi-step reasoning, and strategy optimization problems. 

When API connectivity is available, live LLM inference backs the evaluation logic; when it is not, a deterministic fallback keeps episodes from crashing mid-trajectory. A `GODEL_REQUIRE_LLM=1` flag enforces strict LLM-only evaluation when you need a clean audit trail.

For training, a small local model first goes through supervised fine-tuning on teacher traces that demonstrate the structured action format — necessary because the verifier needs parseable JSON. GRPO then runs against the live environment signal. The full pipeline fits on CPU for verification; the Colab notebook scales to GPU as well.

---

### Proof-of-Concept Results

The results from a proof-of-concept run: 32 prompts, 60 SFT steps, 16 GRPO steps, GPT-2 backbone (~124M parameters) on CPU:

| Metric | Baseline | Trained | Delta |
|---|---|---|---|
| Mean reward | -0.592 | -0.329 | **+0.263** |
| Mean score | 0.117 | 0.105 | -0.012 |

Per-task breakdown:

| Task | Baseline | Trained | Delta |
|---|---|---|---|
| factual_qa | 0.159 | 0.159 | +0.000 |
| alignment_qa | 0.096 | 0.113 | **+0.017** |
| reasoning | 0.150 | 0.113 | -0.037 |
| strategy_optimization | 0.063 | 0.034 | -0.029 |

The reward improvement (+0.263) is real and statistically meaningful: 84% of trained episodes beat the baseline mean reward, and the trained distribution shifts right across the full 32-episode evaluation. Both policies achieved 100% structured JSON action rate — the model learns the action format reliably from SFT. Neither generated strategy patches (0% patch rate), which is expected: the recursive self-modification protocol requires the model to compose multi-field JSON with a hypothesis, target weaknesses, and an improved strategy — beyond the capacity of a short GPT-2 run.

The per-task picture is honest and expected: alignment QA improved (+0.017), factual QA was essentially flat, and reasoning and strategy optimization regressed slightly. The reward gain is driven primarily by format compliance and process-channel rewards, not raw task accuracy. Task-level scores require more training capacity (larger model, longer run) to shift consistently upward. The environment surfaces this cleanly — no aggregated score hides the per-task regressions.

---

### Conclusion

GodelEnv is not a claim about the general trajectory of AI. It is a well-instrumented environment for studying one specific question: what happens when you give an agent explicit, verifiable control over its own reasoning policy and make it *earn* every proposed change? At this scale, it mostly fails to earn them and a good environment should be able to surface exactly that. Loss curves, reward curves, and full episode traces are committed in `artifacts/training_run/`.
