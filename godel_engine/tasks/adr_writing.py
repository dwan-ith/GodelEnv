"""
Architecture Decision Record (ADR) Writing (Hard difficulty).
"""
import random
from typing import Optional, Dict

from godel_engine.tasks.base import BaseTask, TaskInstance

_ADR_DATASET = [
    {
        "id": "adr01",
        "prompt": "Evolve this sparse architecture note into a professional ADR for migrating a monolith to a serverless event-driven architecture.",
        "initial_solution": "Decided to move from monolith to AWS Lambda. Monolith is big and slow. Lambda is cheap and scales. Use SQS for events.",
    }
]

class ADRWritingTask(BaseTask):
    def __init__(self):
        super().__init__("adr_writing", "hard")
        self.dataset = _ADR_DATASET

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
            "structure": "Does the document follow a standard ADR format (Context, Decision, Consequences, Alternatives)?",
            "tradeoff_analysis": "Are both positive and negative consequences thoroughly explained?",
            "consequences": "Does the document explore ripple effects on operations, cost, and developer experience?",
            "clarity": "Is the technical reasoning sound and effectively communicated to stakeholders?"
        }
