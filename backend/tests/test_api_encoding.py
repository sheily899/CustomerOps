import asyncio
from types import SimpleNamespace

import api.main as api_main
from api.main import _ensure_utf8_json_content_type

def test_json_responses_explicitly_advertise_utf8():
    response = SimpleNamespace(headers={"content-type": "application/json"})

    _ensure_utf8_json_content_type(response)

    assert response.headers["content-type"] == "application/json; charset=utf-8"


def test_non_json_content_type_is_unchanged():
    response = SimpleNamespace(headers={"content-type": "text/plain"})

    _ensure_utf8_json_content_type(response)

    assert response.headers["content-type"] == "text/plain"


def test_health_exposes_runtime_knowledge_base_contract(monkeypatch):
    class FakeKnowledgeBase:
        collection_name = "knowledge_base_benchmark_v2"
        embedding_model = "BAAI/bge-small-zh-v1.5"

        async def doc_count_async(self):
            return 120

    monkeypatch.setattr(api_main, "_orchestrator", SimpleNamespace(get_stats=lambda: {}))
    monkeypatch.setattr(api_main, "_knowledge_base", FakeKnowledgeBase())

    payload = asyncio.run(api_main.health())

    assert payload["knowledge_base"] == {
        "collection_name": "knowledge_base_benchmark_v2",
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "doc_count": 120,
    }
