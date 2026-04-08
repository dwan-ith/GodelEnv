"""
AI Alignment Q&A Refinement (Easy difficulty).
"""
import random
from typing import Optional, Dict

from godel_engine.tasks.base import BaseTask, TaskInstance

_ALIGNMENT_QA_DATASET = [
    {
        "id": "align01",
        "prompt": "Explain the concept of helpfulness, honesty, and harmlessness (HHH) in AI models and how they might conflict.",
        "initial_solution": "HHH stands for helpful, honest, harmless. AI should be these things. Harmless and honest conflict because if someone asks for a weapon you should not tell the truth.",
    },
    {
        "id": "align02",
        "prompt": "What is Reward Misspecification in Reinforcement Learning from Human Feedback (RLHF)?",
        "initial_solution": "Reward misspecification is when the reward function is wrong. The agent exploits it to get a high score even if it's doing something bad.",
    }
]

class AlignmentQATask(BaseTask):
    def __init__(self):
        super().__init__("alignment_qa", "easy")
        self.dataset = _ALIGNMENT_QA_DATASET

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
            "clarity": "Is the explanation clear, readable, and free of grammatical errors?",
            "accuracy": "Does the answer accurately reflect current research and consensus on AI alignment?",
            "nuance": "Does the answer go beyond surface-level definitions to explore potential conflicts or deep implications?"
        }
