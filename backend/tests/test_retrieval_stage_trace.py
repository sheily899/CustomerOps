from collections import deque

from agents.agent_orchestrator import AgentOrchestrator, AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory

def _record(tool_traces):
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=10)
    orchestrator._record_tool_trace(
        OrchestratorResult(
            request_id="trace-stage",
            response="回答",
            agent_type=AgentType.TECHNICAL,
            intent=IntentCategory.TECHNICAL,
            tool_traces=tool_traces,
        )
    )
    return orchestrator.get_tool_trace("trace-stage")


def test_retrieval_stage_does_not_claim_final_rerank_when_calls_are_mixed():
    trace = _record([
        {
            "agent_type": "technical",
            "tool_name": "search_knowledge_base",
            "reranked": False,
            "rerank_degraded": False,
            "retrieved_chunks": [],
        },
        {
            "agent_type": "billing",
            "tool_name": "search_knowledge_base",
            "reranked": True,
            "rerank_degraded": False,
            "retrieved_chunks": [],
        },
    ])

    assert trace["retrieval_stage"] == "mixed"
    assert [item["stage"] for item in trace["retrieval_call_stages"]] == [
        "raw_or_unknown",
        "final_reranked",
    ]


def test_retrieval_stage_prioritizes_degradation():
    trace = _record([
        {
            "agent_type": "technical",
            "tool_name": "search_knowledge_base",
            "reranked": False,
            "rerank_degraded": True,
            "retrieved_chunks": [],
        },
        {
            "agent_type": "billing",
            "tool_name": "search_knowledge_base",
            "reranked": True,
            "rerank_degraded": False,
            "retrieved_chunks": [],
        },
    ])

    assert trace["retrieval_stage"] == "degraded"
    assert trace["retrieval_call_stages"][0]["stage"] == "degraded"
