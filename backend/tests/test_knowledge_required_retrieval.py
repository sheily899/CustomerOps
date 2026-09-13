import asyncio
from types import SimpleNamespace

from agents.agent_orchestrator import (
    AgentType,
    BillingAgent,
    TechnicalAgent,
    Request,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel

class TextClient:
    def __init__(self, text="已根据知识库内容回答。"):
        self.calls = []
        self.text = text

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                block = SimpleNamespace(type="text", text=self.text)
                return SimpleNamespace(content=[block])

        self.messages = Messages()


class FailingTextClient:
    def __init__(self):
        class Messages:
            async def create(inner, **kwargs):
                raise RuntimeError("llm unavailable")

        self.messages = Messages()


class RecordingRagManager:
    def __init__(self):
        self.calls = []

    async def search_with_rewrite(self, tool_name, query, top_k=5):
        self.calls.append({"tool_name": tool_name, "query": query, "top_k": top_k})
        return SimpleNamespace(
            success=True,
            data=[
                {
                    "source_chunk_id": "refund_timing#chunk_001",
                    "title": "退款到账时效",
                    "content": "退款审核和到账时间以实际渠道处理为准。",
                    "score": 0.9,
                }
            ],
            reranked=True,
            rerank_degraded=False,
        )


def make_request(**overrides):
    values = {
        "message": "退款多久到账？",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.REFUND,
        "intent_group": "billing",
        "urgency": UrgencyLevel.LOW,
        "intent_confidence": 0.95,
        "knowledge_required": True,
    }
    values.update(overrides)
    return Request(**values)


def test_knowledge_required_forces_one_retrieval_before_agent_answer():
    rag = RecordingRagManager()
    client = TextClient()
    agent = BillingAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(rag))

    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert rag.calls == [
        {"tool_name": "knowledge_search", "query": "退款多久到账？", "top_k": 5}
    ]
    assert response.tools_used == ["search_knowledge_base"]
    assert response.tool_traces[0]["retrieved_chunks"][0]["source_chunk_id"] == (
        "refund_timing#chunk_001"
    )
    assert len(client.calls) == 1
    assert "[系统已完成必需的知识库检索]" in str(client.calls[0]["messages"])


def test_forced_retrieval_trace_survives_llm_failure():
    rag = RecordingRagManager()
    agent = BillingAgent(FailingTextClient(), "test-model")
    agent.set_shared_tools(build_shared_rag_tools(rag))

    response = asyncio.run(agent.handle(make_request()))

    assert response.success is False
    assert response.tool_traces[0]["tool_name"] == "search_knowledge_base"
    assert response.tool_traces[0]["retrieved_chunks"][0]["source_chunk_id"] == (
        "refund_timing#chunk_001"
    )


def test_multi_agent_retrieval_is_independent_per_executed_agent():
    rag = RecordingRagManager()
    shared = build_shared_rag_tools(rag)
    technical = TechnicalAgent(TextClient("技术回答"), "test-model")
    billing = BillingAgent(TextClient("账单回答"), "test-model")
    technical.set_shared_tools(shared)
    billing.set_shared_tools(shared)

    request = make_request(
        message="页面打不开，同时我还想确认支付是否处理中。",
        intent=IntentCategory.TECHNICAL,
        intent_group="technical",
    )
    async def run_both():
        return await asyncio.gather(technical.handle(request), billing.handle(request))

    responses = asyncio.run(run_both())

    assert len(rag.calls) == 2
    assert all(call["tool_name"] == "knowledge_search" for call in rag.calls)
    assert [response.agent_type for response in responses] == [
        AgentType.TECHNICAL,
        AgentType.BILLING,
    ]
    assert all(response.tools_used == ["search_knowledge_base"] for response in responses)
