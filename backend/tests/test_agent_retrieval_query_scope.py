from agents.agent_orchestrator import BillingAgent, TechnicalAgent
from core.intent_recognizer import IntentCategory
from tests.test_agent_orchestrator import FakeClient, make_request

def test_compound_retrieval_query_is_scoped_to_each_agent():
    request = make_request(
        message="应用反复崩溃，我还要索取一张发票，请由两个对应代理分别处理。",
        intent=IntentCategory.TECHNICAL_CRASH,
        secondary_intents=[IntentCategory.INVOICE],
    )

    technical_query = TechnicalAgent(FakeClient(), "test-model")._retrieval_query(request)
    billing_query = BillingAgent(FakeClient(), "test-model")._retrieval_query(request)

    assert technical_query == "应用反复崩溃"
    assert billing_query == "索取一张发票"


def test_single_intent_retrieval_query_keeps_the_original_message():
    request = make_request(
        message="退款多久到账？",
        intent=IntentCategory.REFUND,
        secondary_intents=[],
    )

    query = BillingAgent(FakeClient(), "test-model")._retrieval_query(request)

    assert query == "退款多久到账？"
