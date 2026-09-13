import asyncio
import json
from types import SimpleNamespace

from memory.conversation_memory import MemoryManager, Message, MsgRole

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


class _FakeProfile:
    def __init__(self):
        self.added = []

    def delete(self, **kwargs):
        return None

    def add(self, **kwargs):
        self.added.append(kwargs)


def _build_manager(response):
    manager = MemoryManager.__new__(MemoryManager)
    manager._client = _FakeClient(response)
    manager._model = "test-model"
    manager._profile = _FakeProfile()
    manager._get_working_memory = lambda user_id, conv_id: asyncio.sleep(
        0,
        result=[Message(MsgRole.USER, "我喜欢绿色界面")],
    )
    manager._get_profile = lambda user_id: asyncio.sleep(0, result={})
    return manager


def test_profile_update_uses_json_output_and_validates_fenced_response():
    manager = _build_manager(
        '```json\n{"preferences":["喜欢绿色界面"],"entities":{"问题类型":["界面"]}}\n```'
    )

    asyncio.run(manager.update_profile("u1", "c1"))

    request = manager._client.messages.calls[0]
    assert request["extra_body"] == {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    saved = json.loads(manager._profile.added[0]["documents"][0])
    assert saved["preferences"] == ["喜欢绿色界面"]


def test_profile_validator_rejects_invalid_shape():
    try:
        MemoryManager._validate_profile_data(
            {"preferences": ["偏好"], "entities": {"问题类型": "界面"}}
        )
    except ValueError as exc:
        assert "entities" in str(exc)
    else:
        raise AssertionError("invalid profile shape should be rejected")
