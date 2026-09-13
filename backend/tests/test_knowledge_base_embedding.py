import mcp.knowledge_base as knowledge_base_module
from mcp.knowledge_base import KnowledgeBase

class FakeCollection:
    def __init__(self, name, count_value=0, payload=None):
        self.name = name
        self.count_value = count_value
        self.payload = payload or {"ids": [], "documents": [], "metadatas": []}
        self.embedding_function = None
        self.add_calls = []

    def count(self):
        return self.count_value

    def get(self, include=None):
        return self.payload

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        self.count_value += len(kwargs.get("ids", []))


class FakeClient:
    def __init__(self, target, legacy):
        self.target = target
        self.legacy = legacy

    def heartbeat(self):
        return True

    def get_or_create_collection(self, *, name, metadata, embedding_function):
        assert name == "knowledge_base_zh"
        self.target.embedding_function = embedding_function
        return self.target

    def get_collection(self, *, name):
        assert name == "knowledge_base"
        return self.legacy


def test_chinese_embedding_collection_migrates_legacy_documents(monkeypatch):
    target = FakeCollection("knowledge_base_zh")
    legacy = FakeCollection(
        "knowledge_base",
        count_value=1,
        payload={
            "ids": ["legacy-1"],
            "documents": ["退款规则"],
            "metadatas": [{"title": "退款政策", "chunk_index": 0, "total_chunks": 1}],
        },
    )
    client = FakeClient(target, legacy)
    monkeypatch.setattr(knowledge_base_module.chromadb, "HttpClient", lambda **_: client)
    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model_name: {"model_name": model_name},
    )

    kb = KnowledgeBase()

    assert kb.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert kb.collection_name == "knowledge_base_zh"
    assert target.embedding_function == {"model_name": "BAAI/bge-small-zh-v1.5"}
    assert target.add_calls == [
        {
            "ids": ["legacy-1"],
            "documents": ["退款规则"],
            "metadatas": [{"title": "退款政策", "chunk_index": 0, "total_chunks": 1}],
        }
    ]
