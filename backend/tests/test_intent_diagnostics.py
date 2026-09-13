import asyncio
from collections import deque
from types import SimpleNamespace

from agents.agent_orchestrator import AgentOrchestrator, AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory, IntentRecognizer

class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_intent_recognizer_records_each_source_for_diagnosis():
    recognizer = IntentRecognizer(api_key="test-key")

    async def fake_llm(message, history):
        return {"intent": IntentCategory.HUMAN_HANDOFF, "confidence": 0.82}

    async def fake_embedding(message):
        return {"intent": IntentCategory.PAYMENT_ISSUE, "confidence": 0.76}

    recognizer._llm_recognize = fake_llm
    recognizer._embedding_recognize = fake_embedding
    recognizer._pattern_recognize = lambda message: {
        "intent": IntentCategory.PAYMENT_ISSUE,
        "confidence": 0.5,
    }

    result = asyncio.run(recognizer.recognize("钱还在处理中，算支付成功了吗？"))

    assert result.source_intents == {
        "llm": "human_handoff",
        "embedding": "payment_issue",
        "pattern": "payment_issue",
    }
    assert result.diagnostics["sources"]["llm"]["confidence"] == 0.82
    assert result.diagnostics["final"]["intent"] == result.intent.value


def test_intent_diagnostics_are_preserved_in_tool_trace():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=10)
    diagnostics = {
        "sources": {
            "llm": {"intent": "human_handoff", "confidence": 0.82},
            "embedding": {"intent": "payment_issue", "confidence": 0.76},
            "pattern": {"intent": "payment_issue", "confidence": 0.5},
        },
        "final": {"intent": "human_handoff", "confidence": 0.574},
    }
    result = OrchestratorResult(
        request_id="diagnostic-1",
        response="回答",
        agent_type=AgentType.ESCALATION,
        intent=IntentCategory.HUMAN_HANDOFF,
        intent_diagnostics=diagnostics,
    )

    orchestrator._record_tool_trace(result)

    trace = orchestrator.get_tool_trace("diagnostic-1")
    assert trace["intent_diagnostics"] == diagnostics


def test_llm_failure_rejects_low_confidence_embedding_fallback():
    recognizer = IntentRecognizer(api_key="test-key")

    intent, confidence, _ = recognizer._vote(
        {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True},
        {"intent": IntentCategory.HUMAN_HANDOFF, "confidence": 0.2144},
        {"intent": IntentCategory.REFUND, "confidence": 0.5},
    )

    assert intent == IntentCategory.REFUND
    assert confidence == 0.5


def test_llm_failure_accepts_embedding_fallback_above_threshold():
    recognizer = IntentRecognizer(api_key="test-key")

    intent, confidence, _ = recognizer._vote(
        {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True},
        {"intent": IntentCategory.PAYMENT_ISSUE, "confidence": 0.76},
        {"intent": IntentCategory.QUERY, "confidence": 0.5},
    )

    assert intent == IntentCategory.PAYMENT_ISSUE
    assert confidence == 0.76


def test_llm_failure_prefers_specific_pattern_over_embedding():
    recognizer = IntentRecognizer(api_key="test-key")

    intent, confidence, _ = recognizer._vote(
        {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True},
        {"intent": IntentCategory.HUMAN_HANDOFF, "confidence": 0.82},
        {"intent": IntentCategory.REFUND, "confidence": 0.5},
    )

    assert intent == IntentCategory.REFUND
    assert confidence == 0.5


def test_llm_failure_returns_other_when_only_generic_pattern_and_embedding_is_below_threshold():
    recognizer = IntentRecognizer(api_key="test-key")

    intent, confidence, _ = recognizer._vote(
        {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True},
        {"intent": IntentCategory.PAYMENT_ISSUE, "confidence": 0.44},
        {"intent": IntentCategory.QUERY, "confidence": 0.5},
    )

    assert intent == IntentCategory.OTHER
    assert confidence == 0.0


def test_embedding_fallback_threshold_defaults_to_045():
    recognizer = IntentRecognizer(api_key="test-key")

    assert recognizer.embedding_fallback_threshold == 0.45


def test_llm_intent_uses_forced_tool_call_and_validates_payload():
    response = SimpleNamespace(content=[SimpleNamespace(
        type="tool_use",
        name="classify_intent",
        input={
            "intent": "refund",
            "confidence": 0.98,
            "reasoning": "用户询问退款到账时间",
        },
    )])
    recognizer = IntentRecognizer(api_key="test-key")
    recognizer.client = _FakeClient(response)

    result = asyncio.run(recognizer._llm_recognize("退款多久到账？", None))

    assert result["intent"] == IntentCategory.REFUND
    assert result["confidence"] == 0.98
    assert result["reasoning"] == "用户询问退款到账时间"
    request = recognizer.client.messages.calls[0]
    assert request["tool_choice"] == {
        "type": "tool",
        "name": "classify_intent",
    }
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["tools"][0]["name"] == "classify_intent"
    assert request["tools"][0]["input_schema"]["additionalProperties"] is False


def test_llm_intent_rejects_invalid_tool_payload_as_failure():
    response = SimpleNamespace(content=[SimpleNamespace(
        type="tool_use",
        name="classify_intent",
        input={"intent": "refund", "confidence": 1.4, "reasoning": ""},
    )])
    recognizer = IntentRecognizer(api_key="test-key")
    recognizer.client = _FakeClient(response)

    result = asyncio.run(recognizer._llm_recognize("退款多久到账？", None))

    assert result["failed"] is True
    assert result["intent"] == IntentCategory.OTHER


def test_escalation_is_a_legacy_alias_of_human_handoff():
    assert IntentCategory.ESCALATION is IntentCategory.HUMAN_HANDOFF
    assert IntentCategory.ESCALATION.value == "human_handoff"


def test_legacy_escalation_tool_payload_is_normalized_to_human_handoff():
    response = SimpleNamespace(content=[SimpleNamespace(
        type="tool_use",
        name="classify_intent",
        input={
            "intent": "escalation",
            "confidence": 0.9,
            "reasoning": "用户明确要求人工介入",
        },
    )])
    recognizer = IntentRecognizer(api_key="test-key")
    recognizer.client = _FakeClient(response)

    result = asyncio.run(recognizer._llm_recognize("请转人工", None))

    assert result["intent"] is IntentCategory.HUMAN_HANDOFF
    assert result.get("failed", False) is False


def test_complaint_pattern_does_not_become_human_handoff():
    recognizer = IntentRecognizer(api_key="test-key")

    result = recognizer._pattern_recognize("我只是想投诉这次服务")

    assert result["intent"] is IntentCategory.COMPLAINT
