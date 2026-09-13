import asyncio
from types import SimpleNamespace

from core.intent_recognizer import IntentCategory, IntentRecognizer

class _Messages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _Client:
    def __init__(self, response):
        self.messages = _Messages(response)


def test_intent_prompt_explicitly_separates_request_from_logistics():
    response = SimpleNamespace(content=[SimpleNamespace(
        type="tool_use",
        name="classify_intent",
        input={
            "intent": "request",
            "confidence": 0.95,
            "reasoning": "用户要求修改收货地址",
        },
    )])
    recognizer = IntentRecognizer(api_key="test-key")
    recognizer.client = _Client(response)

    result = asyncio.run(recognizer._llm_recognize("我想改收货地址", None))

    assert result["intent"] is IntentCategory.REQUEST
    prompt = recognizer.client.messages.calls[0]["messages"][0]["content"]
    assert "修改、取消或更新某项内容" in prompt
    assert "只有询问物流状态、配送时效" in prompt
