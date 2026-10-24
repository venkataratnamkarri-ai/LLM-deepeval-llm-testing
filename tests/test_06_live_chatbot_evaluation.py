"""
Test 06: Live Chatbot Multi-Turn Transcript Evaluation
Simulates live, stateful conversations with Cohere chat models maintaining history,
constructs ConversationalTestCase objects from live turns, and evaluates professionalism
using DeepEval ConversationalGEval with custom Cohere judge.
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
from deepeval.test_case import (
    ConversationalTestCase,
    Turn,
    MultiTurnParams,
)
from deepeval.metrics import ConversationalGEval
from src.cohere_evalmodel import CohereModel


def test_live_chatbot_transcripts():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env")

    co = cohere.ClientV2(api_key=api_key)
    judge = CohereModel()

    dialogue_scenarios = [
        [
            "Is Python an interpreted language?",
            "What about C++?",
            "Which one executes faster in benchmarks?"
        ],
        [
            "What was Vasco da Gama famous for?",
            "Who arrived in India first, Columbus or Vasco da Gama?"
        ],
        [
            "What is cloud seeding also known as?",
            "Has India ever attempted artificial rain experiments?"
        ]
    ]

    conversations = []

    for scenario_idx, questions in enumerate(dialogue_scenarios, 1):
        history = []
        turns = []
        print(f"\nSimulating Live Dialogue Scenario {scenario_idx} ({len(questions)} turns)...")

        for question in questions:
            # Format chat history for Cohere ClientV2
            messages = []
            for item in history:
                messages.append({"role": item["role"], "content": item["content"]})
            messages.append({"role": "user", "content": question})

            response = co.chat(
                model="command-a-03-2025",
                messages=messages,
                temperature=0.2,
                max_tokens=150
            )

            answer = "".join(
                item.text for item in response.message.content
                if getattr(item, "text", None)
            ).strip()

            turns.append(Turn(role="user", content=question))
            turns.append(Turn(role="assistant", content=answer))

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        conversations.append(ConversationalTestCase(turns=turns))

    # Metric: Conversational Professionalism
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="The assistant must respond politely, accurately, objectively, and professionally throughout all dialogue turns.",
        evaluation_params=[
            MultiTurnParams.CONTENT,
            MultiTurnParams.ROLE,
        ],
        threshold=0.7,
        model=judge,
    )

    evaluate(
        test_cases=conversations,
        metrics=[professionalism_metric]
    )


if __name__ == "__main__":
    test_live_chatbot_transcripts()
