"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION_NAME = "knowledge_base_zh"
LEGACY_COLLECTION_NAME = "knowledge_base"


def build_embedding_function(model_name: str):
    """构建客户端 Embedding 函数，确保文档和查询使用同一个中文模型。"""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device="cpu",
        normalize_embeddings=True,
    )


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    使用中文 Embedding 模型 BAAI/bge-small-zh-v1.5。
    文档导入和查询都通过同一个客户端 Embedding 函数生成向量，
    避免中文查询与文档向量使用不同模型。
    """

    # 保留类级常量供旧代码读取；实际名称可通过 ECHOMIND_KB_COLLECTION 覆盖。
    COLLECTION_NAME = DEFAULT_COLLECTION_NAME

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        load_defaults: bool = True,
    ):
        self.embedding_model = (
            embedding_model or os.getenv("ECHOMIND_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ).strip()
        self.collection_name = (
            collection_name or os.getenv("ECHOMIND_KB_COLLECTION", DEFAULT_COLLECTION_NAME)
        ).strip()
        self._embedding_function = build_embedding_function(self.embedding_model)

        # 优先连接独立 ChromaDB 服务；Embedding 在应用侧生成后写入/查询服务端。
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 显式传入 Embedding 函数，服务端和本地模式都使用同一个中文模型。
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "EchoMind RAG 知识库",
                "embedding_model": self.embedding_model,
                "hnsw:space": "l2",
            },
            embedding_function=self._embedding_function,
        )

        # 新集合为空时，优先迁移旧集合中的文档并用新模型重新生成向量。
        # 评测集合可显式关闭自动初始化，避免默认文档混入测试语料。
        if load_defaults and self._collection.count() == 0:
            self._migrate_legacy_docs_or_load_defaults()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        可选的 source_document_id / corpus_version 会写入每个 chunk 的 metadata，
        用于评测时跨运行稳定地标识来源。
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            source_document_id = doc.get("source_document_id") or self._slugify(title)
            corpus_version = doc.get("corpus_version") or "kb-v1-default"
            chunks = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append(
                    {
                        "title": title,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_document_id": source_document_id,
                        # 对外稳定 ID 从 001 开始；runtime_id 仍保留 Chroma 的实际 ID。
                        "source_chunk_id": f"{source_document_id}#chunk_{i + 1:03d}",
                        "content_hash": f"sha256:{hashlib.sha256(chunk.encode()).hexdigest()}",
                        "corpus_version": corpus_version,
                    }
                )

        if ids:
            # 由显式配置的中文 Embedding 函数生成向量。
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, Any]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """导入已经冻结的 chunk，并保留其 source_chunk_id。"""
        ids, docs, metas = [], [], []

        for chunk in chunks:
            source_chunk_id = str(chunk.get("source_chunk_id") or "").strip()
            content = str(chunk.get("content") or "")
            if not source_chunk_id or not content:
                raise ValueError("冻结 chunk 必须包含 source_chunk_id 和 content")

            source_document_id = str(
                chunk.get("source_document_id")
                or source_chunk_id.split("#chunk_", 1)[0]
            )
            suffix = source_chunk_id.rsplit("#chunk_", 1)[-1]
            try:
                chunk_index = int(suffix) - 1 if suffix.isdigit() else int(chunk.get("chunk_index", 0))
            except (TypeError, ValueError):
                chunk_index = 0
            content_hash = str(chunk.get("content_hash") or "")
            if not content_hash:
                content_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
            corpus_version = str(chunk.get("corpus_version") or "kb-v1-default")
            runtime_id = hashlib.md5(
                f"{source_chunk_id}\n{content_hash}".encode()
            ).hexdigest()

            ids.append(runtime_id)
            docs.append(content)
            metas.append(
                {
                    "title": str(chunk.get("title") or ""),
                    "chunk_index": chunk_index,
                    "total_chunks": int(chunk.get("total_chunks") or 1),
                    "source_document_id": source_document_id,
                    "source_chunk_id": source_chunk_id,
                    "content_hash": content_hash,
                    "corpus_version": corpus_version,
                }
            )

        if ids:
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info("知识库导入 %s 个冻结文档片段", len(ids))
        return len(ids)

    async def add_chunks_async(self, chunks: List[Dict[str, Any]]) -> int:
        """异步导入已经冻结的 chunk。"""
        return await asyncio.to_thread(self.add_chunks, chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 使用同一个中文 Embedding 函数将 query 转为向量，再进行 L2 检索。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]
            if not ids:
                ids = [""] * len(documents)
            for runtime_id, doc, meta, dist in zip(
                ids,
                documents,
                metadatas,
                distances,
            ):
                meta = meta or {}
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                    "runtime_id": runtime_id,
                    "source_document_id": meta.get("source_document_id", ""),
                    "source_chunk_id": meta.get("source_chunk_id", ""),
                    "content_hash": meta.get("content_hash", ""),
                    "corpus_version": meta.get("corpus_version", ""),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _slugify(title: str) -> str:
        """为未显式提供来源 ID 的旧调用生成可读的稳定兜底 ID。"""
        import re

        value = re.sub(r"[^\w\u4e00-\u9fff]+", "_", title.strip())
        return value.strip("_") or "doc"

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")

    def _migrate_legacy_docs_or_load_defaults(self) -> None:
        """将旧默认集合的文档迁移到中文集合，避免切换模型后丢失已有知识。"""
        try:
            legacy = self._client.get_collection(name=LEGACY_COLLECTION_NAME)
            legacy_data = legacy.get(include=["documents", "metadatas"])
            ids = legacy_data.get("ids", [])
            documents = legacy_data.get("documents", [])
            metadatas = legacy_data.get("metadatas", [])
            if ids and documents:
                payload: Dict[str, Any] = {"ids": ids, "documents": documents}
                if metadatas:
                    payload["metadatas"] = metadatas
                self._collection.add(**payload)
                logger.info(
                    "已将旧知识库迁移到中文 Embedding 集合: %s -> %s，共 %s 个文档片段",
                    LEGACY_COLLECTION_NAME,
                    self.collection_name,
                    len(documents),
                )
                return
        except Exception as exc:
            logger.info("未找到可迁移的旧知识库，使用默认文档初始化: %s", exc)

        self._load_default_docs()
