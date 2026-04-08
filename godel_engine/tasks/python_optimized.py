"""
Python Script Optimization & Documentation (Medium difficulty).
"""
import random
from typing import Optional, Dict

from godel_engine.tasks.base import BaseTask, TaskInstance

_PYTHON_DATASET = [
    {
        "id": "py01",
        "prompt": "Optimize this Python script that finds prime numbers and add comprehensive docstrings and type hints.",
        "initial_solution": "import math\ndef get_primes(n):\n    primes = []\n    for i in range(2, n):\n        is_prime = True\n        for j in range(2, i):\n            if i % j == 0:\n                is_prime = False\n                break\n        if is_prime:\n            primes.append(i)\n    return primes\n\nprint(get_primes(100))",
    }
]

class PythonOptimizedTask(BaseTask):
    def __init__(self):
        super().__init__("python_optimized", "medium")
        self.dataset = _PYTHON_DATASET

    def sample(self, rng: Optional[random.Random] = None) -> TaskInstance:
        _rng = rng or random.Random()
        data = _rng.choice(self.dataset)
        return TaskInstance(
            task_id=data["id"],
            difficulty=self.difficulty,
            prompt=data["prompt"],
            initial_solution=data["initial_solution"],
            reference=data
        )

    def _get_rubrics(self) -> Dict[str, str]:
        return {
            "efficiency": "Is the code optimized for performance (e.g., O(n log log n) or better)?",
            "documentation": "Are there comprehensive docstrings and type hints for all functions?",
            "robustness": "Does the code handle edge cases like n < 2 gracefully?"
        }
