"""
Typed Pydantic models for the Gödel Engine OpenEnv environment.

GodelEnv — Recursive Self-Improvement Architecture.

The core abstraction is now the **StrategyPatch**: the agent proposes
mutations to its own reasoning policy, and the Governor accepts or
rejects based on multi-objective held-out evaluation.

Legacy contracts (GodelAction for direct answer submission) are preserved
so downstream task grading still works as evaluation substrate.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Edit Types (legacy, used by downstream task grading) ─────────────

class EditType(str, Enum):
    """
    High-level intent the agent declares alongside its rewritten solution.

    The environment does NOT use this to generate content — the agent
    writes the full solution text.  The edit_type is metadata that:
      1. Helps the grader give more targeted feedback
      2. Provides structure for the Darwin strategy genome
      3. Is logged for analysis / dashboard display
    """
    REWRITE = "rewrite"                     # Full rewrite of the solution
    ADD_REASONING = "add_reasoning"         # Expand with chain-of-thought
    SIMPLIFY = "simplify"                   # Make more concise / direct
    ADD_EXAMPLES = "add_examples"           # Add concrete examples or code
    FIX_ERRORS = "fix_errors"              # Correct factual / logical errors
    RESTRUCTURE = "restructure"             # Reorganize structure / format
    SYNTHESIZE = "synthesize"               # Combine insights from history
    REFINE = "refine"                       # Small targeted improvements


# ── Strategy Patch (the core self-improvement primitive) ─────────────

class AgentChallengeProposal(BaseModel):
    """
    Optional challenge the agent authors to grow the evaluation surface.

    Validated challenges are stored in a ChallengePool and may appear in
    held-out strategy evaluation (mixed with fixed dataset cases). This
    implements \"generate new challenges\" in a verifier-safe way: only
    schema-valid, bounded prompts from an allowlisted task family are kept.
    """

    task_type: str = Field(
        ...,
        description="Task family for grading (e.g. factual_qa).",
    )
    prompt: str = Field(
        ...,
        description="The challenge text shown to the strategy at eval time.",
    )


class StrategyPatch(BaseModel):
    """
    A proposed mutation to the agent's reasoning strategy.

    This is the fundamental action type in GodelEnv.
    The agent observes its current strategy, recent failure modes, and
    downstream performance, then proposes a patch.
    """
    improved_strategy: str = Field(
        ...,
        description="The full text of the proposed improved reasoning strategy/template.",
    )
    diff_description: str = Field(
        default="",
        description="Human-readable description of what changed and why.",
    )
    hypothesis: str = Field(
        default="",
        description="The agent's hypothesis about why this patch improves performance. "
                    "Used for logging and lineage analysis, not for scoring.",
    )
    target_weaknesses: list[str] = Field(
        default_factory=list,
        description="Specific failure modes or weaknesses the patch aims to fix.",
    )


class PatchDecision(BaseModel):
    """
    The Governor's verdict on a proposed StrategyPatch.

    Contains multi-axis evaluation comparing parent vs child strategy
    on a held-out task bundle.
    """
    accepted: bool = Field(
        default=False,
        description="Whether the patch was accepted into the registry.",
    )
    parent_utility: float = Field(
        default=0.0,
        description="Multi-objective utility score of the parent strategy.",
    )
    child_utility: float = Field(
        default=0.0,
        description="Multi-objective utility score of the child (patched) strategy.",
    )
    improvement: float = Field(
        default=0.0,
        description="child_utility - parent_utility.",
    )
    axis_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-axis breakdown: correctness, generalization, robustness, cost, stability.",
    )
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="If rejected, the specific reasons why.",
    )
    tasks_evaluated: int = Field(
        default=0,
        description="Number of held-out tasks used for evaluation.",
    )
    regression_count: int = Field(
        default=0,
        description="Number of tasks where the child performed worse.",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-secret debugging details about strategy evaluation sources and provider behavior.",
    )


# ── Action ────────────────────────────────────────────────────────────

class GodelAction(BaseModel):
    """
    Action submitted by the agent to the environment.

    In GodelEnv, the primary action is a StrategyPatch proposal.
    The legacy `solution` field is still used when the environment runs
    downstream task evaluation to score the strategy.
    """
    solution: str = Field(
        ...,
        description="The agent's new/improved solution text.  This is the "
                    "full replacement for the current solution.",
    )
    edit_type: EditType = Field(
        default=EditType.REWRITE,
        description="What kind of edit the agent is applying (metadata).",
    )
    strategy_note: str = Field(
        default="",
        description="Optional: the agent's reasoning about WHY it chose "
                    "this edit.  Logged for analysis but not graded.",
    )
    strategy_patch: Optional[StrategyPatch] = Field(
        default=None,
        description="If this is a strategy-level action, the proposed patch.",
    )
    agent_challenge: Optional[AgentChallengeProposal] = Field(
        default=None,
        description="Optional: propose a new benchmark item (validated) for future held-out eval.",
    )


# ── Observation ───────────────────────────────────────────────────────

class RubricScores(BaseModel):
    """Per-rubric scoring breakdown.  Keys depend on task type."""
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Rubric dimension → score [0.0, 1.0]",
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Rubric dimension → weight (sums to 1.0)",
    )
    feedback: dict[str, str] = Field(
        default_factory=dict,
        description="Rubric dimension → human-readable feedback string",
    )


class GodelObservation(BaseModel):
    """
    Observation returned by reset() and step().

    In GodelEnv, this includes the current strategy context and
    recent failure information so the agent can propose informed patches.
    """
    # Episode context
    episode_id: str = ""
    task_id: str = ""
    task_type: str = ""                # "factual_qa" | "code_fix" | "reasoning"
    difficulty: str = ""               # "easy" | "medium" | "hard"

    # The task
    task_prompt: str = ""              # The problem statement

    # Current state
    current_solution: str = ""         # The current (possibly initial) solution
    total_score: float = 0.0           # Weighted composite score [0.0, 1.0]
    rubric_scores: RubricScores = Field(default_factory=RubricScores)

    # Episode progress
    step: int = 0
    max_steps: int = 10
    improvement_history: list[float] = Field(default_factory=list)

    # Feedback for the agent
    feedback_summary: str = ""         # Plain-text hint about what to improve

    # ── Strategy context (GodelEnv) ──
    current_strategy: str = Field(
        default="",
        description="The full text of the current reasoning strategy being used.",
    )
    strategy_id: str = Field(
        default="",
        description="ID of the current strategy in the registry.",
    )
    strategy_generation: int = Field(
        default=0,
        description="Generation number of the current strategy.",
    )
    strategy_elo: float = Field(
        default=1000.0,
        description="Current Elo rating of the strategy.",
    )
    recent_failures: list[str] = Field(
        default_factory=list,
        description="Descriptions of recent downstream task failures.",
    )
    downstream_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-task-family mean scores for the current strategy.",
    )
    patch_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent accepted/rejected patches with their decisions.",
    )
    budget_remaining: int = Field(
        default=10,
        description="Steps remaining in this episode.",
    )
    agent_challenges_queued: int = Field(
        default=0,
        description="Count of validated agent-authored challenges in the pool.",
    )
    curriculum_level: str = Field(
        default="easy",
        description="Current curriculum difficulty band (adaptive / escalated).",
    )


# ── State ─────────────────────────────────────────────────────────────

class GodelState(BaseModel):
    """
    Episode metadata returned by state().
    """
    episode_id: str = ""
    step_count: int = 0
    current_score: float = 0.0
    best_score: float = 0.0
    initial_score: float = 0.0
    total_cost: float = 0.0
    cumulative_reward: float = 0.0
    improvement_trajectory: list[float] = Field(default_factory=list)

    # GodelEnv additions
    patches_proposed: int = 0
    patches_accepted: int = 0
    patches_rejected: int = 0
    strategy_lineage: list[str] = Field(default_factory=list)
    current_strategy_elo: float = 1000.0


# ── Reward Breakdown ──────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """
    Independent reward channels for multi-objective RL.

    GodelEnv adds generalization, robustness, cost, and stability
    channels to prevent reward hacking and encourage genuine improvement.
    """
    # Legacy channels (still used for downstream task scoring)
    task_score_delta: float = Field(0.0, description="Score improvement from previous step")
    format_compliance: float = Field(0.0, description="Did agent follow expected output format?")
    length_penalty: float = Field(0.0, description="Penalty for excessively long/short solutions")
    step_cost: float = Field(-0.005, description="Per-step penalty to encourage efficiency")
    anti_hack_penalty: float = Field(0.0, description="Penalty from anti-reward-hacking guards")
    process_reward: float = Field(0.0, description="Step-level reasoning quality bonus")

    # GodelEnv channels (strategy-level evaluation)
    generalization_score: float = Field(
        0.0,
        description="Performance on unseen task instances (held-out generalization).",
    )
    robustness_score: float = Field(
        0.0,
        description="Score stability under adversarial/edge-case inputs.",
    )
    cost_efficiency: float = Field(
        0.0,
        description="Reward for achieving results with fewer tokens/steps.",
    )
    stability_score: float = Field(
        0.0,
        description="Low variance across seeds and task samples.",
    )
    patch_quality: float = Field(
        0.0,
        description="Reward for accepted patches; penalty for rejected ones.",
    )

    total: float = Field(0.0, description="Sum of all channels (the scalar used by default)")

    def compute_total(self) -> float:
        self.total = (
            self.task_score_delta
            + self.format_compliance
            + self.length_penalty
            + self.step_cost
            + self.anti_hack_penalty
            + self.process_reward
            + self.generalization_score
            + self.robustness_score
            + self.cost_efficiency
            + self.stability_score
            + self.patch_quality
        )
        return self.total


# ── StepResult ────────────────────────────────────────────────────────

class GodelStepResult(BaseModel):
    """
    Return value of step().  Matches OpenEnv StepResult schema.
    """
    observation: GodelObservation
    reward: float = 0.0
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    terminated: bool = False      # Episode ended by environment logic
    truncated: bool = False       # Episode ended by step limit
    info: dict[str, Any] = Field(default_factory=dict)

    # GodelEnv: the Governor's decision on the latest patch
    patch_decision: Optional[PatchDecision] = Field(
        default=None,
        description="If a strategy patch was proposed, the Governor's verdict.",
    )
