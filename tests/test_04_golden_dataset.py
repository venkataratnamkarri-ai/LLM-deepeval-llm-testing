"""
Test 04: Golden Dataset Benchmark Evaluation
Demonstrates creating an EvaluationDataset composed of verified 'Golden' reference samples
and evaluating batch test cases against baseline ground truth outputs.
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

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden
from src.cohere_evalmodel import CohereModel


def test_golden_dataset_benchmark():
    judge = CohereModel()

    correctness_metric = GEval(
        name="Correctness",
        criteria="The model's response should be factually correct and relevant to the question.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=judge,
    )

    # 1. Define Golden Benchmark Records
    goldens = [
        Golden(
            input="What is the capital of France?",
            expected_output="The capital of France is Paris."
        ),
        Golden(
            input="What is the value of pi?",
            expected_output="The value of pi is approximately 3.14159."
        ),
        Golden(
            input="What does CPU stand for in computer science?",
            expected_output="CPU stands for Central Processing Unit."
        )
    ]

    dataset = EvaluationDataset(goldens=goldens)

    # 2. Generate model responses for each golden
    for golden in dataset.goldens:
        actual_answer = judge.generate(golden.input)
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                actual_output=actual_answer,
                expected_output=golden.expected_output,
            )
        )

    # 3. Batch evaluation
    evaluate(
        test_cases=dataset.test_cases,
        metrics=[correctness_metric],
    )


if __name__ == "__main__":
    test_golden_dataset_benchmark()
