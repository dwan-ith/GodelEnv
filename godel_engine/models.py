"""
Typed Pydantic models for the Gödel Engine OpenEnv environment.

These are the contract between agent and environment:
  - GodelAction:       what the agent submits each step
  - GodelObservation:  what the environment returns
  - GodelState:        episode metadata
  - GodelStepResult:   step() return value (observation + reward + signals)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Action ────────────────────────────────────────────────────────────

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


class GodelAction(BaseModel):
    """
    Action submitted by the agent to the environment.

    The agent WRITES the improved solution text.  The environment
    grades it — it does not generate content.
    """
    solution: str = Field(
        ...,
        description="The agent's new/improved solution text.  This is the "
                    "full replacement for the current solution."
    )
    edit_type: EditType = Field(
        default=EditType.REWRITE,
        description="What kind of edit the agent is applying (metadata)."
    )
    strategy_note: str = Field(
        default="",
        description="Optional: the agent's reasoning about WHY it chose "
                    "this edit.  Logged for analysis but not graded."
    )


# ── Observation ───────────────────────────────────────────────────────

class RubricScores(BaseModel):
    """Per-rubric scoring breakdown.  Keys depend on task type."""
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Rubric dimension → score [0.0, 1.0]"
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Rubric dimension → weight (sums to 1.0)"
    )
    feedback: dict[str, str] = Field(
        default_factory=dict,
        description="Rubric dimension → human-readable feedback string"
    )


class GodelObservation(BaseModel):
    """
    Observation returned by reset() and step().

    Contains everything the agent needs to decide its next action.
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


# ── StepResult ────────────────────────────────────────────────────────

class GodelStepResult(BaseModel):
    """
    Return value of step().  Matches OpenEnv StepResult schema.
    """
    observation: GodelObservation
    reward: float = 0.0
    terminated: bool = False      # Episode ended by environment logic
    truncated: bool = False       # Episode ended by step limit
    info: dict[str, Any] = Field(default_factory=dict)
