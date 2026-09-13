import asyncio
from collections import deque

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentType,
    OrchestratorResult,
    Request,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel

def _recognizer_with_primary(primary: IntentCategory) -> IntentRecognizer:
    recognizer = IntentRecognizer(api_key="test-key")

    async def fake_llm(message, history):
        return {"intent": primary, "confidence": 0.95, "reasoning": "test"}

    async def fake_embedding(message):
        return {"intent": primary, "confidence": 0.9}

    recognizer._llm_recognize = fake_llm
    recognizer._embedding_recognize = fake_embedding
    recognizer._pattern_recognize = lambda message: {
        "intent": primary,
        "confidence": 0.8,
    }
    return recognizer


def test_single_intent_keeps_secondary_intents_empty():
    recognizer = _recognizer_with_primary(IntentCategory.REFUND)

    result = asyncio.run(recognizer.recognize("退款多久到账？"))

    assert result.secondary_intents == []


def test_explicit_compound_request_detects_payment_issue_secondary_intent():
    recognizer = _recognizer_with_primary(IntentCategory.TECHNICAL)

    result = asyncio.run(
        recognizer.recognize("页面打不开，同时我还想确认一笔支付是否处于处理中，请分别处理这两个问题。")
    )

    assert result.intent is IntentCategory.TECHNICAL
    assert result.secondary_intents == [IntentCategory.PAYMENT_ISSUE]


def test_compound_request_keeps_clear_llm_primary_instead_of_other():
    recognizer = IntentRecognizer(api_key="test-key")

    async def fake_llm(message, history):
        return {
            "intent": IntentCategory.TECHNICAL,
            "confidence": 0.6,
            "reasoning": "页面问题是主要诉求",
        }

    async def fake_embedding(message):
        return {"intent": IntentCategory.COMPLAINT, "confidence": 0.2}

    recognizer._llm_recognize = fake_llm
    recognizer._embedding_recognize = fake_embedding
    recognizer._pattern_recognize = lambda message: {
        "intent": IntentCategory.TECHNICAL,
        "confidence": 0.5,
    }

    result = asyncio.run(
        recognizer.recognize("页面打不开，同时我还想确认一笔支付是否处于处理中。")
    )

    assert result.intent is IntentCategory.TECHNICAL
    assert result.secondary_intents == [IntentCategory.PAYMENT_ISSUE]


def test_compound_request_does_not_duplicate_generic_billing_intent():
    recognizer = _recognizer_with_primary(IntentCategory.TECHNICAL_LOGIN)

    result = asyncio.run(
        recognizer.recognize("我登录失败了，还想申请一笔退款，请让技术和账单分别核验。")
    )

    assert result.intent is IntentCategory.TECHNICAL_LOGIN
    assert result.secondary_intents == [IntentCategory.REFUND]


def test_handoff_request_does_not_create_secondary_agent_intent():
    recognizer = _recognizer_with_primary(IntentCategory.HUMAN_HANDOFF)

    result = asyncio.run(
        recognizer.recognize("退款问题我要求直接升级人工，请保留订单和交易背景。")
    )

    assert result.intent is IntentCategory.HUMAN_HANDOFF
    assert result.secondary_intents == []


def test_network_request_failure_is_a_technical_pattern():
    recognizer = IntentRecognizer(api_key="test-key")

    result = recognizer._pattern_recognize("网络请求失败")

    assert result["intent"] is IntentCategory.TECHNICAL


def test_related_words_without_independent_request_are_not_split():
    recognizer = _recognizer_with_primary(IntentCategory.REFUND)

    result = asyncio.run(recognizer.recognize("退款页面打不开"))

    assert result.secondary_intents == []


def test_compound_intent_routes_to_secondary_domain_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
    }
    request = Request(
        message="页面打不开，同时我还想确认一笔支付是否处于处理中。",
        user_id="u1",
        conv_id="c1",
        intent=IntentCategory.TECHNICAL,
        intent_group="technical",
        urgency=UrgencyLevel.LOW,
        intent_confidence=0.9,
        secondary_intents=[IntentCategory.PAYMENT_ISSUE],
    )

    decision = orchestrator._route_decision(request)

    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]


def test_handoff_route_is_terminal_and_does_not_start_supporting_agents():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
        AgentType.ESCALATION: [object()],
    }
    request = Request(
        message="退款问题我要求直接升级人工，请保留订单和交易背景。",
        user_id="u1",
        conv_id="c1",
        intent=IntentCategory.HUMAN_HANDOFF,
        intent_group="human_handoff",
        urgency=UrgencyLevel.HIGH,
        intent_confidence=0.9,
        secondary_intents=[IntentCategory.REFUND],
    )

    decision = orchestrator._route_decision(request)

    assert decision.primary_agent is AgentType.ESCALATION
    assert decision.supporting_agents == []
    assert not decision.multi_agent


def test_compound_intent_is_preserved_in_trace():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=10)
    result = OrchestratorResult(
        request_id="trace-compound",
        response="已分别处理。",
        agent_type=AgentType.TECHNICAL,
        intent=IntentCategory.TECHNICAL,
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
        secondary_intents=[IntentCategory.PAYMENT_ISSUE],
    )

    orchestrator._record_tool_trace(result)

    trace = orchestrator.get_tool_trace("trace-compound")
    assert trace["secondary_intents"] == ["payment_issue"]


def test_trace_records_executed_agents():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=10)
    result = OrchestratorResult(
        request_id="trace-agents",
        response="已分别处理。",
        agent_type=AgentType.TECHNICAL,
        intent=IntentCategory.TECHNICAL,
        agent_types=[AgentType.TECHNICAL, AgentType.BILLING],
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
    )

    orchestrator._record_tool_trace(result)

    trace = orchestrator.get_tool_trace("trace-agents")
    assert trace["executed_agents"] == ["technical", "billing"]
