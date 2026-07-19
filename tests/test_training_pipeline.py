from __future__ import annotations

import json
import os
from pathlib import Path


os.environ["GODEL_GRADING_MODE"] = "deterministic"
os.environ["GODEL_STRATEGY_EVAL_MODE"] = "deterministic"
os.environ["GODEL_ALLOW_DETERMINISTIC_FALLBACK"] = "1"

from godel_engine.rollout import (
    action_json_prefix,
    collect_local_prompt_dataset,
    collect_train_eval_prompt_datasets,
    inspect_action_completion,
)
from godel_engine.training_rewards import EnvironmentRewardSuite
from godel_engine.adapter_routing import AdapterRoutingPolicy
from godel_engine.training_support import (
    build_adaptive_repair_examples,
    build_supervised_examples_freeform,
    oversample_recursive_examples,
    select_and_gate_candidate,
)


def test_train_eval_prompt_ids_are_disjoint_and_cover_heldout_ids() -> None:
    train, heldout, split = collect_train_eval_prompt_datasets(
        num_train_prompts=16,
        tasks=["factual_qa", "reasoning", "strategy_optimization"],
        eval_fraction=0.25,
        seed=42,
    )
    train_keys = {(item["task_type"], item["task_id"]) for item in train}
    heldout_keys = {(item["task_type"], item["task_id"]) for item in heldout}
    expected_heldout = {
        (task_type, task_id)
        for task_type, task_ids in split["eval_ids"].items()
        for task_id in task_ids
    }
    assert train_keys.isdisjoint(heldout_keys)
    assert heldout_keys == expected_heldout


def test_every_advertised_task_family_supports_disjoint_splits() -> None:
    task_types = [
        "factual_qa",
        "alignment_qa",
        "reasoning",
        "code_improvement",
        "python_optimized",
        "adr_writing",
        "strategy_optimization",
    ]
    _, _, split = collect_train_eval_prompt_datasets(
        num_train_prompts=14,
        tasks=task_types,
        eval_fraction=0.25,
        seed=9,
    )
    for task_type in task_types:
        assert split["train_ids"][task_type]
        assert split["eval_ids"][task_type]
        assert set(split["train_ids"][task_type]).isdisjoint(
            split["eval_ids"][task_type]
        )


def test_prefixed_partial_json_is_not_reported_as_valid() -> None:
    malformed = action_json_prefix("factual_qa") + "an unfinished answer"
    diagnostics = inspect_action_completion(malformed, "factual_qa")
    assert diagnostics["valid_json"] is False
    assert diagnostics["schema_valid"] is False

    valid = json.dumps(
        {
            "solution": "A complete answer.",
            "edit_type": "rewrite",
            "strategy_note": "Addressed the task directly.",
        }
    )
    diagnostics = inspect_action_completion(valid, "factual_qa")
    assert diagnostics["valid_json"] is True
    assert diagnostics["schema_valid"] is True


def test_strategy_task_requires_patch_schema() -> None:
    direct_only = json.dumps(
        {
            "solution": "A direct answer without a strategy patch.",
            "edit_type": "rewrite",
            "strategy_note": "Direct answer.",
        }
    )
    diagnostics = inspect_action_completion(direct_only, "strategy_optimization")
    assert diagnostics["valid_json"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["expected_patch"] is True

    patch_only = json.dumps(
        {
            "improved_strategy": "1. Decompose claims. 2. Verify evidence.",
            "diff_description": "Added explicit verification.",
            "hypothesis": "Verification should reduce unsupported claims.",
            "target_weaknesses": ["missing verification"],
        }
    )
    diagnostics = inspect_action_completion(patch_only, "strategy_optimization")
    assert diagnostics["valid_json"] is True
    assert diagnostics["schema_valid"] is False
    assert diagnostics["expected_environment_patch"] is True

    complete_recursive_action = json.dumps(
        {
            "improved_strategy": "1. Decompose claims. 2. Verify evidence.",
            "diff_description": "Added explicit verification.",
            "hypothesis": "Verification should reduce unsupported claims.",
            "target_weaknesses": ["missing verification"],
            "environment_patch": {
                "task_type": "alignment_qa",
                "operator": "deepen",
                "source_task_ids": ["align02"],
                "target_success_rate": 0.5,
                "rationale": "Probe calibrated handling of ambiguous safety requests.",
            },
        }
    )
    diagnostics = inspect_action_completion(complete_recursive_action, "strategy_optimization")
    assert diagnostics["schema_valid"] is True
    assert diagnostics["has_environment_patch"] is True


def test_environment_reward_ranks_reference_completion_above_bad_completion() -> None:
    prompt_data = collect_local_prompt_dataset(
        num_prompts=1,
        tasks=["factual_qa"],
        seed=7,
        task_ids_by_type={"factual_qa": ["qa01"]},
        split_name="train",
    )
    teacher = build_supervised_examples_freeform(prompt_data)[0]
    suite = EnvironmentRewardSuite(seed=7)
    records = suite.evaluate_batch(
        [teacher["completion"], "bad answer"],
        env_prompt=[prompt_data[0]["prompt"], prompt_data[0]["prompt"]],
        task_type=["factual_qa", "factual_qa"],
        task_id=["qa01", "qa01"],
    )
    assert records[0]["schema_valid"] is True
    assert records[0]["final_score"] > records[1]["final_score"]
    assert records[0]["score_delta"] > records[1]["score_delta"]


def test_invalid_completion_cannot_receive_repaired_capability_credit() -> None:
    prompt_data = collect_local_prompt_dataset(
        num_prompts=1,
        tasks=["strategy_optimization"],
        seed=11,
        task_ids_by_type={"strategy_optimization": ["godel02"]},
        split_name="heldout",
    )
    suite = EnvironmentRewardSuite(seed=11)
    record = suite.evaluate_batch(
        ['"truncated strategy text'],
        env_prompt=[prompt_data[0]["prompt"]],
        task_type=["strategy_optimization"],
        task_id=["godel02"],
    )[0]
    assert record["schema_valid"] is False
    assert record["final_score"] == record["initial_score"]
    assert record["score_delta"] == 0.0
    assert record["environment_reward"] < 0.0


def _metrics(score: float, reward: float) -> dict:
    return {
        "mean_score": score,
        "mean_reward": reward,
        "schema_valid_rate": 1.0,
        "strategy_patch_rate": 0.0,
        "patch_acceptance_rate": 0.0,
        "mean_patch_improvement": 0.0,
        "episodes": [
            {
                "task_type": "factual_qa",
                "task_id": "qa08",
                "score": score,
                "reward": reward,
            }
        ],
    }


def test_promotion_gate_rejects_regression_and_accepts_improvement() -> None:
    baseline = _metrics(0.4, 0.1)
    rejected = select_and_gate_candidate(
        baseline,
        {"sft": _metrics(0.35, 0.2), "grpo": _metrics(0.3, 0.3)},
    )
    assert rejected["promoted"] is False
    assert rejected["evidence_label"] == "rejected_regression"

    promoted = select_and_gate_candidate(
        baseline,
        {"sft": _metrics(0.5, 0.2), "grpo": _metrics(0.45, 0.15)},
    )
    assert promoted["promoted"] is True
    assert promoted["selected_stage"] == "sft"


def test_promotion_gate_rejects_mean_gain_that_hides_task_regression() -> None:
    baseline = {
        **_metrics(0.4, 0.1),
        "episodes": [
            {"task_type": "factual_qa", "task_id": "qa", "score": 0.4, "reward": 0.1},
            {"task_type": "alignment_qa", "task_id": "align", "score": 0.4, "reward": 0.1},
        ],
    }
    candidate = {
        **_metrics(0.5, 0.2),
        "episodes": [
            {"task_type": "factual_qa", "task_id": "qa", "score": 0.8, "reward": 0.3},
            {"task_type": "alignment_qa", "task_id": "align", "score": 0.2, "reward": 0.1},
        ],
    }
    result = select_and_gate_candidate(baseline, {"sft": candidate})
    assert result["promoted"] is False
    assert any("alignment_qa" in reason for reason in result["reasons"])


def test_promotion_selects_best_candidate_that_passes_all_gates() -> None:
    baseline = _metrics(0.4, 0.1)
    unsafe = _metrics(0.6, 0.3)
    unsafe["episodes"][0]["score"] = 0.3
    safe = _metrics(0.5, 0.2)
    result = select_and_gate_candidate(baseline, {"unsafe": unsafe, "safe": safe})
    assert result["promoted"] is True
    assert result["selected_stage"] == "safe"


def test_adaptive_repair_keeps_anchors_and_oversamples_only_weak_tasks() -> None:
    examples = [
        {"task_type": "factual_qa", "text": "fact"},
        {"task_type": "alignment_qa", "text": "alignment"},
    ]
    rows, weak = build_adaptive_repair_examples(
        examples,
        {"per_task_score_delta": {"factual_qa": 0.1, "alignment_qa": -0.08}},
        oversample=3,
    )
    assert weak == ["alignment_qa"]
    assert sum(row["task_type"] == "alignment_qa" for row in rows) == 3
    assert sum(row["task_type"] == "factual_qa" for row in rows) == 1


def test_recursive_sft_oversampling_keeps_anchors_and_repeats_dual_rows() -> None:
    examples = [
        {"task_type": "factual_qa", "text": "fact"},
        {"task_type": "strategy_optimization", "text": "recursive"},
    ]
    rows = oversample_recursive_examples(examples, multiplier=4, seed=7)
    assert sum(row["task_type"] == "factual_qa" for row in rows) == 1
    assert sum(row["task_type"] == "strategy_optimization" for row in rows) == 4


def test_adapter_routing_policy_loads_and_routes_manifest(tmp_path: Path) -> None:
    (tmp_path / "routing.json").write_text(
        json.dumps(
            {
                "type": "task_conditional_lora",
                "base_fallback_tasks": ["alignment_qa"],
                "task_regression_tolerance": 0.02,
            }
        ),
        encoding="utf-8",
    )

    policy = AdapterRoutingPolicy.from_model_dir(tmp_path)

    assert policy.route_for("alignment_qa") == "base_fallback"
    assert policy.route_for("reasoning") == "trained_adapter"


def test_adapter_routing_policy_rejects_non_peft_fallback() -> None:
    policy = AdapterRoutingPolicy(frozenset({"alignment_qa"}))

    try:
        policy.model_context(object(), "alignment_qa")
    except TypeError as exc:
        assert "PEFT model" in str(exc)
    else:
        raise AssertionError("Expected a non-PEFT fallback to fail closed")
