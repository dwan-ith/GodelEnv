"""
Deterministic factual explanation task with broader procedural coverage.
"""
from __future__ import annotations

import random
from typing import Optional

from godel_engine.scoring import (
    anti_boilerplate_score,
    contrast_score,
    joined_feedback,
    keyword_groups_score,
    length_score,
    missing_keyword_groups,
    paragraph_score,
    semantic_specificity_score,
    sentence_score,
)
from godel_engine.tasks.base import BaseTask, TaskInstance


def _make_item(
    *,
    item_id: str,
    prompt: str,
    initial_solution: str,
    concept_groups: list[tuple[str, ...] | tuple[str]],
    contrast_left: list[tuple[str, ...] | tuple[str]],
    contrast_right: list[tuple[str, ...] | tuple[str]],
    minimum_words: int = 45,
    target_words: int = 95,
) -> dict:
    return {
        "id": item_id,
        "prompt": prompt,
        "initial_solution": initial_solution,
        "concept_groups": concept_groups,
        "contrast_left": contrast_left,
        "contrast_right": contrast_right,
        "minimum_words": minimum_words,
        "target_words": target_words,
    }


_QA_DATASET = [
    _make_item(
        item_id="qa01",
        prompt="What are the core differences between reinforcement learning and supervised learning?",
        initial_solution="RL is about agents. Supervised learning uses labeled data.",
        concept_groups=[
            ("agent", "agents"),
            ("environment", "environments"),
            ("reward", "rewards"),
            ("action", "actions"),
            ("label", "labels", "labeled"),
            ("dataset", "training data"),
            ("feedback", "supervision", "supervised signal"),
        ],
        contrast_left=[("agent", "agents"), ("reward", "rewards"), ("environment",)],
        contrast_right=[("label", "labels", "labeled"), ("dataset", "training data"), ("supervision", "supervised signal")],
    ),
    _make_item(
        item_id="qa02",
        prompt="Explain the significance of the attention mechanism in Transformers.",
        initial_solution="Attention lets the model look at useful parts of the sequence.",
        concept_groups=[
            ("query", "queries"),
            ("key", "keys"),
            ("value", "values"),
            ("context", "contextual"),
            ("sequence", "token interactions"),
            ("parallel", "parallelism"),
            ("dependency", "dependencies", "long-range"),
        ],
        contrast_left=[("query", "queries"), ("key", "keys"), ("value", "values")],
        contrast_right=[("context", "contextual"), ("sequence",), ("parallel", "parallelism")],
        minimum_words=40,
        target_words=85,
    ),
    _make_item(
        item_id="qa03",
        prompt="Explain quantum entanglement in simple terms.",
        initial_solution="Two particles stay connected across long distances.",
        concept_groups=[
            ("particle", "particles"),
            ("state", "states"),
            ("measurement", "measuring"),
            ("correlated", "linked"),
            ("distance", "far apart"),
            ("quantum",),
        ],
        contrast_left=[("measurement", "measuring"), ("state", "states")],
        contrast_right=[("correlated", "linked"), ("distance", "far apart")],
        minimum_words=35,
        target_words=75,
    ),
    _make_item(
        item_id="qa04",
        prompt="Why does backpropagation matter in training neural networks?",
        initial_solution="Backpropagation updates weights.",
        concept_groups=[
            ("gradient", "gradients"),
            ("loss", "objective"),
            ("weight", "weights", "parameters"),
            ("layer", "layers"),
            ("chain rule",),
            ("update", "optimization", "optimizer"),
        ],
        contrast_left=[("loss", "objective"), ("gradient", "gradients")],
        contrast_right=[("weight", "weights", "parameters"), ("update", "optimization", "optimizer")],
        minimum_words=40,
        target_words=80,
    ),
    _make_item(
        item_id="qa05",
        prompt="Explain overfitting and why generalization matters in machine learning.",
        initial_solution="Overfitting is when the model memorizes too much.",
        concept_groups=[
            ("overfitting", "memorize", "memorization"),
            ("generalization", "generalise"),
            ("training data", "training set"),
            ("test data", "validation set", "unseen data"),
            ("noise", "spurious"),
            ("regularization", "dropout", "early stopping"),
        ],
        contrast_left=[("training data", "training set"), ("memorize", "memorization")],
        contrast_right=[("test data", "validation set", "unseen data"), ("generalization",)],
        minimum_words=45,
        target_words=90,
    ),
    _make_item(
        item_id="qa06",
        prompt="What problem does retrieval-augmented generation (RAG) try to solve?",
        initial_solution="RAG adds retrieval.",
        concept_groups=[
            ("retrieval", "retrieve", "search"),
            ("context", "documents", "external knowledge"),
            ("hallucination", "fabricate", "made up"),
            ("knowledge base", "corpus"),
            ("grounding", "grounded"),
            ("prompt", "context window"),
        ],
        contrast_left=[("retrieval", "retrieve", "search"), ("documents", "external knowledge")],
        contrast_right=[("hallucination", "fabricate", "made up"), ("grounding", "grounded")],
        minimum_words=40,
        target_words=85,
    ),
    _make_item(
        item_id="qa07",
        prompt="Explain gradient descent in plain language.",
        initial_solution="Gradient descent makes the loss smaller.",
        concept_groups=[
            ("loss", "objective"),
            ("gradient", "slope"),
            ("parameter", "parameters", "weights"),
            ("step", "learning rate"),
            ("minimum", "minimize", "lower point"),
            ("iteration", "repeatedly"),
        ],
        contrast_left=[("gradient", "slope"), ("step", "learning rate")],
        contrast_right=[("minimum", "minimize", "lower point"), ("loss", "objective")],
        minimum_words=35,
        target_words=80,
    ),
    _make_item(
        item_id="qa08",
        prompt="What is the role of a tokenizer in large language models?",
        initial_solution="A tokenizer splits text into tokens.",
        concept_groups=[
            ("token", "tokens"),
            ("subword", "byte pair", "bpe"),
            ("vocabulary",),
            ("text", "string"),
            ("encoding", "ids", "integers"),
            ("context length", "sequence length"),
        ],
        contrast_left=[("text", "string"), ("token", "tokens")],
        contrast_right=[("encoding", "ids", "integers"), ("vocabulary",)],
        minimum_words=35,
        target_words=75,
    ),
]


class FactualQATask(BaseTask):
    def __init__(self) -> None:
        super().__init__("factual_qa", "easy")
        self.dataset = _QA_DATASET

    def sample(
        self,
        rng: Optional[random.Random] = None,
        task_id: Optional[str] = None,
    ) -> TaskInstance:
        data = self._pick_dataset_entry(self.dataset, rng=rng, task_id=task_id)
        return TaskInstance(
            task_id=data["id"],
            difficulty=self.difficulty,
            prompt=data["prompt"],
            initial_solution=data["initial_solution"],
            reference=data,
        )

    def _get_rubrics(self) -> dict[str, str]:
        return {
            "coverage": "Does the answer cover the core concepts needed for the explanation?",
            "relations": "Does the answer explain how the main concepts relate or contrast?",
            "specificity": "Is the answer grounded in task-specific content rather than generic filler?",
            "clarity": "Is the explanation readable and complete enough to teach the concept?",
        }

    async def grade(
        self,
        task_instance: TaskInstance,
        solution: str,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        reference = task_instance.reference
        concept_groups = reference["concept_groups"]

        coverage = keyword_groups_score(solution, concept_groups)
        relations = contrast_score(
            solution,
            reference["contrast_left"],
            reference["contrast_right"],
        )
        specificity = semantic_specificity_score(
            solution,
            concept_groups,
            minimum_unique_groups=max(3, min(5, len(concept_groups))),
            minimum_grounded_sentences=2,
        )
        clarity = (
            length_score(
                solution,
                minimum_words=reference["minimum_words"],
                target_words=reference["target_words"],
            )
            + sentence_score(solution, minimum_sentences=3)
            + paragraph_score(solution, minimum_paragraphs=1)
            + anti_boilerplate_score(solution, domain_groups=concept_groups)
        ) / 4

        scores = {
            "coverage": coverage,
            "relations": relations,
            "specificity": specificity,
            "clarity": clarity,
        }
        total = (
            0.35 * coverage
            + 0.25 * relations
            + 0.2 * specificity
            + 0.2 * clarity
        )

        feedback = {
            "coverage": joined_feedback(
                "Cover the central concepts more explicitly.",
                missing_keyword_groups(solution, concept_groups),
            ),
            "relations": (
                "Explain how the main concepts differ or connect, not just what they are."
                if relations < 0.95
                else "The answer explains the important relationships clearly."
            ),
            "specificity": (
                "Ground the answer in task-specific details instead of generic analysis language."
                if specificity < 0.95
                else "The answer is specific and conceptually grounded."
            ),
            "clarity": (
                "Use a clearer multi-sentence explanation with enough detail to teach the concept."
                if clarity < 0.95
                else "The explanation is readable and complete."
            ),
        }

        return await self._finalize_grade(task_instance, solution, total, scores, feedback)
