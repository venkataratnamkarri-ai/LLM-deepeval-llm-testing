"""
Sanity verification script to validate that all required LLM testing libraries
(DeepEval, Cohere, Pydantic, json_repair, pytest) are properly installed.
"""

import deepeval
import cohere
import pydantic
import json_repair
import pytest

def test_environment_packages():
    print("\n" + "=" * 50)
    print("Environment Verification:")
    print(f"  • DeepEval:    {deepeval.__version__}")
    print(f"  • Cohere SDK:  {cohere.__version__}")
    print(f"  • Pydantic:    {pydantic.__version__}")
    print(f"  • Pytest:      {pytest.__version__}")
    print("=" * 50)
    assert True

if __name__ == "__main__":
    test_environment_packages()
