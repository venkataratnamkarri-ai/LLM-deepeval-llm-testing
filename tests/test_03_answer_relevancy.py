"""
Test 03: Answer Relevancy Metric Evaluation
Validates that generated responses are relevant and directly address the user's question
without including redundant or non-sequitur statements.
"""

import os
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

import cohere
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from src.cohere_evalmodel import CohereModel


def test_answer_relevancy():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env")

    co = cohere.ClientV2(api_key=api_key)

    question = "What is the difference between an interpreted and a compiled programming language?"

    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=200
    )

    answer = "".join(
        item.text for item in response.message.content
        if getattr(item, "text", None)
    ).strip()

    judge = CohereModel()

    relevance_metric = AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True,
        model=judge,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
    )

    evaluate(
        test_cases=[test_case],
        metrics=[relevance_metric]
    )


if __name__ == "__main__":
    test_answer_relevancy()
