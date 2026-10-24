"""
Test 01: Faithfulness Metric Evaluation
Validates whether the model's actual output is faithful to (grounded in)
the supplied retrieval context. Uses custom CohereModel as the evaluator judge.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure root directory is on sys.path for src imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from src.cohere_evalmodel import CohereModel


def test_faithfulness_metric():
    judge = CohereModel()

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True,
        model=judge,
    )

    # Test case where actual_output contradicts the retrieval context
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=[
            "The capital of France is Madrid."
        ]
    )

    evaluate(
        test_cases=[test_case],
        metrics=[faithfulness_metric]
    )


if __name__ == "__main__":
    test_faithfulness_metric()
