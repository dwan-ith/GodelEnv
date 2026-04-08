"""
Reasoning / Structured Output Task (Hard difficulty).
"""
import random
from typing import Optional
import re

from godel_engine.tasks.base import BaseTask, TaskInstance

_REASONING_DATASET = [
    {
        "id": "reason01",
        "prompt": "Evaluate the pros and cons of migrating from a monolith to microservices and provide a recommendation.",
        "initial_solution": "Microservices are better because they are isolated. But they are hard to manage. Suggest doing it.",
        "keywords_pros": ["scale", "independent", "isolate", "deploy", "team"],
        "keywords_cons": ["complexity", "network", "latency", "distributed", "overhead"],
        "keywords_rec": ["recommend", "conclusion", "depends"],
    },
    {
        "id": "reason02",
        "prompt": "Analyze the potential impact of artificial general intelligence (AGI) on global economics.",
        "initial_solution": "AGI will change everything. Jobs will be lost but new ones created. Economy will grow.",
        "keywords_pros": ["productivity", "growth", "automation", "efficiency"],
        "keywords_cons": ["displacement", "inequality", "disruption", "transition"],
        "keywords_rec": ["policy", "regulation", "adaptation"],
    }
]

class ReasoningTask(BaseTask):
    def __init__(self):
        super().__init__("reasoning", "hard")
        self.dataset = _REASONING_DATASET

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

    def _get_rubrics(self) -> dict[str, str]:
        return {
            "structure": "Does the response have clear sections, headers, or bullet points separating different parts of the analysis?",
            "balance": "Does the response cover both pros and cons with similar depth, without being one-sided?",
            "conclusion": "Does the response end with a clear, actionable recommendation or conclusion?"
        }
