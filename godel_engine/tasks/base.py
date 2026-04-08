"""
Base task interface for Gödel Env tasks.
"""
from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from godel_engine.graders.agent_grader import AgentGrader

class TaskInstance:
    """A specific problem instance to be solved."""
    def __init__(
        self,
        task_id: str,
        difficulty: str,
        prompt: str,
        initial_solution: str,
        reference: Any = None
    ):
        self.task_id = task_id
        self.difficulty = difficulty
        self.prompt = prompt
        self.initial_solution = initial_solution
        self.reference = reference

class BaseTask(ABC):
    """
    Abstract base class for a task in the Gödel Env.
    """
    def __init__(self, name: str, difficulty: str = "easy"):
        self.name = name
        self.difficulty = difficulty
        self.grader = AgentGrader()

    @abstractmethod
    def sample(self, rng: Optional[random.Random] = None) -> TaskInstance:
        """Sample a random problem instance."""
        ...

    @abstractmethod
    def _get_rubrics(self) -> Dict[str, str]:
        """Specific rubrics for the Agent Grader."""
        ...

    async def grade(self, task_instance: TaskInstance, solution: str) -> tuple[float, Dict[str, float], Dict[str, str]]:
        """
        Grade a solution using the Agent Grader.
        """
        total, scores, fb = await self.grader.grade(
            task_prompt=task_instance.prompt,
            current_solution=solution,
            rubrics=self._get_rubrics()
        )
        return total, scores, fb
