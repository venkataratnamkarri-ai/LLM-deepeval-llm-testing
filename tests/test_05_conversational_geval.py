"""
Test 05: Multi-Turn Conversational G-Eval Evaluation
Demonstrates evaluating conversational agent transcripts over multiple turns
for qualitative behavioral criteria (e.g. Professionalism, Tone, Politeness).
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
from deepeval.test_case import (
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
)
from deepeval.metrics import ConversationalGEval
from src.cohere_evalmodel import CohereModel


def test_conversational_professionalism():
    judge = CohereModel()

    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="The assistant should respond professionally, politely, and respectfully throughout the conversation.",
        evaluation_params=[
            MultiTurnParams.CONTENT,
            MultiTurnParams.ROLE,
        ],
        threshold=0.7,
        model=judge,
    )

    # Conversation 1: Unprofessional/Snarky Assistant (Expected to Fail threshold)
    unprofessional_conv = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Is Python an interpreted language?"),
            Turn(role="assistant", content="Of course, how could you not know that?"),
            Turn(role="user", content="What about C++?"),
            Turn(role="assistant", content="Seriously? C++ is compiled, read a book!"),
        ]
    )

    # Conversation 2: Professional & Courteous Assistant (Expected to Pass threshold)
    professional_conv = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Do you have feelings and emotions?"),
            Turn(role="assistant", content="I don't experience feelings or emotions, but I can discuss them empathetically and assist you."),
            Turn(role="user", content="Should I wear a white or navy shirt with beige trousers?"),
            Turn(role="assistant", content="A crisp white shirt offers a clean, versatile look, while navy blue provides a sharp, formal contrast. Both pair nicely."),
        ]
    )

    evaluate(
        test_cases=[unprofessional_conv, professional_conv],
        metrics=[professionalism_metric],
    )


if __name__ == "__main__":
    test_conversational_professionalism()
