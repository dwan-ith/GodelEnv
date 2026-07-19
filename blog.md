# GodelEnv: The Model Improves, Then the World Moves

Most LLM reinforcement learning optimizes answers against a benchmark that never changes. GodelEnv asks a harder question: can the model improve its own reasoning procedure while the environment keeps generating the next useful challenge?

Every recursive action contains two proposals. A `StrategyPatch` rewrites the reusable reasoning policy. An `EnvironmentPatch` selects a bounded mutation of verified tasks. Two independent Governors decide what survives. The strategy Governor replays parent and child on hidden tasks and rejects weak gains, broad regressions, catastrophic failures, instability, canary failures, and evaluator leakage. The environment Governor owns the generated challenge and hidden reference, then checks novelty, teacher solvability, teacher-current regret, capability-frontier proximity, and total learning value. The LLM is never allowed to rewrite its tests.

That distinction matters because an earlier training run made the model worse while increasing aggregate reward. The repaired pipeline uses disjoint task IDs, paired held-out evaluation, six live environment-backed TRL rewards, completion-only LoRA SFT, weak-family repair, bootstrap intervals, and per-family promotion gates. Invalid JSON receives no repaired capability credit, and a checkpoint is not deployed merely because its average reward rose.

The committed Qwen 0.5B experiment improved held-out mean score from `0.4356` to `0.4510` (`+0.0155`, paired 95% CI `[-0.0185, +0.0617]`). Environment reward rose from `-0.0589` to `0.0962` (`+0.1551`, 95% CI `[+0.0095, +0.3217]`), and strict schema validity rose from 50% to 87.5%. Raw adapters damaged factual and alignment behavior, so the promoted routed policy keeps untouched base weights for those families.

We report recursive evidence separately. A clean deterministic mechanism run proposed ten model and ten environment mutations, accepted two strategy descendants, admitted three frontier challenges, and persisted both lineages. A strict local `gemma4:e2b` smoke produced both mutation types and used the LLM for all recorded hidden strategy evaluations; both proposals were rejected by their Governors, demonstrating that hybrid execution does not bypass verification. An earlier hybrid smoke separately contains an accepted LLM strategy mutation.

The promoted policy also crossed both recursive gates on held-out episodes: one strategy mutation was accepted with positive hidden-replay improvement, and one generated challenge was admitted with positive learning value. The run is therefore labeled `verified_coevolution`. The evidence boundary remains explicit: this is a bounded proof on two held-out strategy episodes, not a claim of open-ended autonomous self-improvement. The deterministic run separately proves repeated coupled mechanics, and the hybrid runs prove the live LLM integration.

The goal is not an agent that declares itself better. It is a system where model improvement and environment improvement are both inspectable, replayable, and difficult to fake.

- [Live environment](https://huggingface.co/spaces/litterarum/GodelEnv)
- [Training notebook](train_colab.ipynb)
- [Training metrics and plots](artifacts/training_run/metrics.json)
- [Coevolution metrics](artifacts/coevolution_smoke_v2/metrics.json)
- [Hybrid dual-action smoke](artifacts/hybrid_coevolution_smoke.json)
