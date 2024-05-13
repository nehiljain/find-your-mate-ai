from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.find_your_mate_ai.config import settings
import os


def test_case():
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra costs."
        ],
    )
    assert_test(test_case, [answer_relevancy_metric])
