"""
Factual Q&A Task (Easy difficulty).
"""
import random
from typing import Optional

from godel_engine.tasks.base import BaseTask, TaskInstance

_QA_DATASET = [
    {
        "id": "qa01",
        "prompt": "What are the core differences between reinforcement learning and supervised learning?",
        "initial_solution": "RL is about agents. Supervised uses data.",
        "keywords": ["agent", "environment", "reward", "action", "label", "data", "feedback"],
        "min_words": 15,
    },
    {
        "id": "qa02",
        "prompt": "Explain the significance of the attention mechanism in Transformers.",
        "initial_solution": "It looks at parts of the sentence.",
        "keywords": ["context", "weight", "queries", "keys", "values", "sequence", "parallel"],
        "min_words": 15,
    },
    {
        "id": "qa03",
        "prompt": "Explain quantum entanglement in simple terms.",
        "initial_solution": "Two particles are connected over long distances.",
        "keywords": ["particle", "state", "linked", "instantaneous", "distance", "measurement"],
        "min_words": 15,
    }
]

class FactualQATask(BaseTask):
    def __init__(self):
        super().__init__("factual_qa", "easy")
        self.dataset = _QA_DATASET

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
            "coverage": "Does the answer cover the key scientific or factual concepts completely?",
            "detail": "Is the explanation sufficiently detailed and not just a single sentence?",
            "structure": "Is the explanation well-structured and easy to read?"
        }
