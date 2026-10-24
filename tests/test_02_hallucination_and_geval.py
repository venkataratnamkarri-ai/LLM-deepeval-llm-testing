"""
Test 02: Hallucination & Custom G-Eval Metric Evaluation
Validates that LLM responses are free of hallucinations and satisfy custom
correctness criteria using DeepEval assert_test and CohereModel judge.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, HallucinationMetric
from src.cohere_evalmodel import CohereModel


def test_hallucination_and_correctness():
    judge = CohereModel()

    # 1. Custom Criteria G-Eval Metric
    correctness_metric = GEval(
        name="Correctness",
        criteria="The model's response should be factually correct and relevant to the question.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=judge
    )

    # 2. Hallucination Metric
    hallucination_metric = HallucinationMetric(
        threshold=0.7,
        model=judge
    )

    question = "How many colors are there in a rainbow?"
    answer = judge.generate(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output="There are seven colors in a rainbow.",
        context=[
            "A rainbow consists of seven colors: red, orange, yellow, green, blue, indigo, and violet."
        ],
    )

    assert_test(
        test_case,
        [correctness_metric, hallucination_metric],
    )

    for metric in [correctness_metric, hallucination_metric]:
        print(f"\nMetric: {metric.__class__.__name__}")
        print("Score:", metric.score)
        print("Success:", metric.success)
        print("Reason:", getattr(metric, "reason", "N/A"))


if __name__ == "__main__":
    test_hallucination_and_correctness()
