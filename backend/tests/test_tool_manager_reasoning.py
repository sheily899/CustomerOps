import asyncio
from types import SimpleNamespace

from mcp.tool_manager import MCPToolManager, ToolResult

class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, dict):
            return SimpleNamespace(content=[self.response])
        return SimpleNamespace(content=[{"type": "text", "text": self.response}])


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def build_manager(response):
    manager = MCPToolManager("test-key", model="test-model")
    manager._client = FakeClient(response)
    return manager


def test_query_rewrite_disables_thinking_mode():
    manager = build_manager('{"queries":["退款审核时效"]}')

    asyncio.run(manager.rewrite_query("退款多久到账？"))

    assert manager._client.messages.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


def test_query_rewrite_rejects_invalid_json_shape():
    manager = build_manager('{"queries": ["退款审核时效", 42]}')

    result = asyncio.run(manager.rewrite_query("退款多久到账？"))

    assert result == ["退款多久到账？"]


def test_query_rewrite_accepts_json_output_wrapped_in_markdown_fence():
    manager = build_manager('```json\n{"queries":["退款审核时效"]}\n```')

    result = asyncio.run(manager.rewrite_query("退款多久到账？"))

    assert result == ["退款多久到账？", "退款审核时效"]


def test_rerank_disables_thinking_mode():
    manager = build_manager({
        "type": "tool_use",
        "name": "rerank_results",
        "input": {"order": [1, 0]},
    })
    items = [{"title": "A", "content": "a"}, {"title": "B", "content": "b"}]

    result = asyncio.run(manager._rerank("问题", items, top_k=1))

    assert result == [items[1]]
    assert manager._client.messages.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
    assert manager._client.messages.calls[0]["tool_choice"] == {
        "type": "tool",
        "name": "rerank_results",
    }


def test_rerank_failure_has_explicit_degradation_status():
    manager = build_manager("not-json")
    items = [{"title": "A"}, {"title": "B"}]

    result, degraded = asyncio.run(manager._rerank_with_status("问题", items, top_k=1))

    assert result == [items[0]]
    assert degraded is True


def test_rerank_rejects_incomplete_or_out_of_range_order():
    manager = build_manager({
        "type": "tool_use",
        "name": "rerank_results",
        "input": {"order": [2, 0]},
    })
    items = [{"title": "A"}, {"title": "B"}]

    result, degraded = asyncio.run(manager._rerank_with_status("问题", items, top_k=1))

    assert result == [items[0]]
    assert degraded is True


def test_search_with_rewrite_deduplicates_by_source_chunk_id():
    manager = build_manager("[]")
    rerank_input = {}

    async def fake_rewrite(query, n=3):
        return ["原始问题", "改写问题"]

    async def fake_call(name, params, context=None, *, use_cache=True, rerank_top_k=0):
        if params["query"] == "原始问题":
            data = [
                {"source_chunk_id": "chunk-1", "score": 0.2, "content": "相同内容"},
                {"source_chunk_id": "chunk-2", "score": 0.4, "content": "其他内容"},
            ]
        else:
            data = [
                {"source_chunk_id": "chunk-1", "score": 0.9, "content": "相同内容"},
            ]
        return ToolResult(success=True, data=data, tool_name=name)

    async def fake_rerank_with_status(query, items, top_k):
        rerank_input["items"] = items
        return items[:top_k], False

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call
    manager._rerank_with_status = fake_rerank_with_status

    result = asyncio.run(
        manager.search_with_rewrite("search_knowledge_base", "原始问题", top_k=5)
    )

    assert [item["source_chunk_id"] for item in rerank_input["items"]] == [
        "chunk-1",
        "chunk-2",
    ]
    assert rerank_input["items"][0]["score"] == 0.9
    assert [item["source_chunk_id"] for item in result.data] == ["chunk-1", "chunk-2"]
