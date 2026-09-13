import asyncio
from types import SimpleNamespace

from evaluation.evaluator import EndToEndEvaluator, EvalResult, LLMJudge

class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[{"type": "text", "text": self.response}])


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_llm_judge_failures_are_not_mixed_into_quality_average():
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._history = []
    evaluator._baseline = None
    evaluator._baseline_path = None

    async def fake_dialog_case(case, case_idx):
        return [
            EvalResult(
                test_id="judge-failed",
                passed=False,
                scores={"relevance": 0.5, "accuracy": 0.5},
                metadata={"judge_failed": True},
            )
        ]

    evaluator._evaluate_dialog_case = fake_dialog_case

    report = asyncio.run(evaluator.run(dialog_cases=[{"question": "问题"}]))

    assert report.avg_scores == {}
    assert report.pass_rate == 0.0


def test_llm_judge_uses_json_output_and_validates_all_scores():
    judge = LLMJudge(_FakeClient(
        '{"relevance":0.9,"accuracy":0.8,"completeness":0.7,"helpfulness":0.6}'
    ), "test-model")

    scores = asyncio.run(judge.judge("问题", "回答"))

    assert scores.judge_failed is False
    assert scores.relevance == 0.9
    assert scores.overall == 0.75
    request = judge._client.messages.calls[0]
    assert request["extra_body"] == {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


def test_llm_judge_rejects_missing_or_out_of_range_scores():
    judge = LLMJudge(_FakeClient(
        '{"relevance":0.9,"accuracy":0.8,"completeness":2.0}'
    ), "test-model")

    scores = asyncio.run(judge.judge("问题", "回答"))

    assert scores.judge_failed is True
    assert scores.error


def test_llm_judge_accepts_json_output_wrapped_in_markdown_fence():
    judge = LLMJudge(_FakeClient(
        '```json\n{"relevance":0.9,"accuracy":0.8,"completeness":0.7,"helpfulness":0.6}\n```'
    ), "test-model")

    scores = asyncio.run(judge.judge("问题", "回答"))

    assert scores.judge_failed is False
    assert scores.overall == 0.75
