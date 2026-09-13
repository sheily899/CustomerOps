"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    ORDER_STATUS = "order_status"        # 订单状态
    LOGISTICS = "logistics"              # 物流配送
    REFUND = "refund"                    # 退款/退货
    INVOICE = "invoice"                  # 发票
    PAYMENT_ISSUE = "payment_issue"      # 支付/扣款异常
    ACCOUNT_SECURITY = "account_security" # 账户安全
    TECHNICAL_LOGIN = "technical_login"  # 登录认证故障
    TECHNICAL_CRASH = "technical_crash"  # 崩溃/错误码
    HUMAN_HANDOFF = "human_handoff"      # 转人工
    ESCALATION = "human_handoff"         # 旧版意图值的兼容别名
    OTHER      = "other"


_INTENT_ALIASES = {
    "escalation": "human_handoff",
}


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    source_intents: Dict[str, str] = field(default_factory=dict)
    # 默认为空；只有独立复合诉求通过第二阶段规则时才填充。
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    compound_trigger: str = ""

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """返回仅用于诊断的三路识别结果，不参与路由或投票。"""
        source_names = ("llm", "embedding", "pattern")
        sources = {
            name: {
                "intent": self.source_intents.get(name, IntentCategory.OTHER.value),
                "confidence": round(float(self.source_scores.get(name, 0.0) or 0.0), 4),
            }
            for name in source_names
        }
        extra_scores = {
            key: round(float(value), 4)
            for key, value in self.source_scores.items()
            if key not in source_names
        }
        payload: Dict[str, Any] = {
            "sources": sources,
            "final": {
                "intent": self.intent.value,
                "confidence": round(float(self.confidence), 4),
            },
            "compound": {
                "detected": bool(self.secondary_intents),
                "secondary_intents": [intent.value for intent in self.secondary_intents],
                "trigger": self.compound_trigger,
            },
        }
        if extra_scores:
            payload["adjustments"] = extra_scores
        return payload


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.ORDER_STATUS: ["我的订单现在是什么状态？", "订单有没有发货？", "订单处理到哪一步了？"],
    IntentCategory.LOGISTICS: ["快递什么时候到？", "物流一直不更新", "配送要多久？"],
    IntentCategory.REFUND: ["我要申请退款", "退货退款怎么处理？", "退款多久到账？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么重复扣款？", "支付失败怎么办？", "这个月多扣了钱"],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请升级处理"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}

_INTENT_TOOL_NAME = "classify_intent"
_INTENT_TOOL = {
    "name": _INTENT_TOOL_NAME,
    "description": "根据用户消息返回一个客服意图及其置信度。",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [category.value for category in IntentCategory],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["intent", "confidence"],
        "additionalProperties": False,
    },
}

# 复合请求只在出现明确的分句/连接信号时才进入第二阶段判断。
# 单个词同时出现并不等于两个独立诉求，例如“退款页面打不开”仍可只是
# 退款流程中的一个技术现象，不能仅凭关键词把它拆成两个意图。
_COMPOUND_STRONG_TRIGGER_RE = re.compile(
    r"(?:同时|另外|还想|还要|而且|并且|此外|分别|两个问题|两个诉求|除此之外|但是|不过|以及)"
)
_HANDOFF_TRIGGER_RE = re.compile(
    r"(?:转人工|人工客服|人工处理|人工介入|升级人工|找人工)"
)
_COMPOUND_CLAUSE_SPLIT_RE = re.compile(
    r"(?:同时|另外|还想|还要|而且|并且|此外|分别|两个问题|两个诉求|除此之外|但是|不过|以及|[，,；;\n。！？!?])"
)

# 这里只用于复合检测，不改变现有三路主意图投票的关键词口径。
# 每个候选意图只承担一个可独立识别的业务主题。
_COMPOUND_PATTERNS: Dict[IntentCategory, List[str]] = {
    IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "人工处理", "人工介入", "升级人工", "找人工"],
    IntentCategory.ORDER_STATUS: ["订单状态", "订单进展", "处理到哪", "订单有没有发货"],
    IntentCategory.LOGISTICS: ["物流", "快递", "配送", "运单"],
    IntentCategory.REFUND: ["退款", "退货", "refund", "return"],
    IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
    IntentCategory.PAYMENT_ISSUE: [
        "重复扣款", "多扣", "支付失败", "支付处理中", "支付是否处于处理中",
        "支付结果", "支付状态", "扣费", "交易未确认", "payment failed"
    ],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗", "账号被盗", "异常登录", "重置密码", "两步验证"],
    IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "401", "验证码"],
    IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "500", "报错", "crash"],
    IntentCategory.TECHNICAL: [
        "页面打不开", "页面无法打开", "页面加载", "加载不出来", "网络请求失败", "网络失败", "访问页面", "403"
    ],
    IntentCategory.BILLING: ["账单", "金额核对", "账单金额"],
}


def _intent_domain(intent: IntentCategory) -> str:
    """返回用于判断跨域协作的稳定业务域，不改变原有 intent_group。"""
    if intent in {
        IntentCategory.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN,
        IntentCategory.TECHNICAL_CRASH,
    }:
        return "technical"
    if intent in {
        IntentCategory.BILLING,
        IntentCategory.ACCOUNT,
        IntentCategory.ACCOUNT_SECURITY,
        IntentCategory.REFUND,
        IntentCategory.INVOICE,
        IntentCategory.PAYMENT_ISSUE,
    }:
        return "billing"
    if intent == IntentCategory.HUMAN_HANDOFF:
        return "escalation"
    return "general"


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
        embedding_fallback_threshold: float = 0.45,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # LLM 识别失败时，只有达到该阈值的 Embedding 结果才允许接管。
        # 具体细粒度 Pattern 会优先于 Embedding；低分结果最终回到 OTHER，
        # 避免低置信度的向量或宽泛关键词直接改变路由。
        self.embedding_fallback_threshold = max(
            0.0,
            min(1.0, float(embedding_fallback_threshold)),
        )
        # 本地字符 n-gram 向量始终可用；如果未来客户端暴露 embeddings 资源，
        # _embed_text 会优先尝试远端向量，否则自动回退本地向量。
        self._embedding_enabled = True

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        source_intents = {
            "llm": self._intent_value(llm.get("intent")),
            "embedding": self._intent_value(emb.get("intent")),
            "pattern": self._intent_value(pat.get("intent")),
        }
        # 明确的 500/服务不可用属于崩溃类故障；避免 LLM/embedding 的泛化结果
        # 将它降级成笼统的 technical，保证技术故障能够进入对应排障流程。
        normalized_message = message.lower()
        if "500" in normalized_message and intent != IntentCategory.TECHNICAL_LOGIN:
            intent = IntentCategory.TECHNICAL_CRASH
            confidence = max(confidence, 0.8)
            source_scores["rule_override"] = 1.0
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)
        secondary_intents, compound_trigger = self._detect_secondary_intents(message, intent)
        # 复合请求中，多个来源分摊置信度后可能刚好低于总阈值，
        # 将本来清晰的 LLM 主意图错误降级为 OTHER。只在已经通过独立
        # 复合触发规则、且 LLM 自身达到阈值时恢复主意图；单意图路径不受影响。
        if (
            intent == IntentCategory.OTHER
            and secondary_intents
            and not llm.get("failed")
            and llm.get("intent") != IntentCategory.OTHER
            and float(llm.get("confidence", 0.0) or 0.0) >= self.threshold
        ):
            intent = llm["intent"]
            confidence = max(confidence, float(llm.get("confidence", 0.0) or 0.0))
            source_scores["compound_primary_rescue"] = float(llm.get("confidence", 0.0) or 0.0)
            secondary_intents, compound_trigger = self._detect_secondary_intents(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
            source_intents=source_intents,
            secondary_intents=secondary_intents,
            compound_trigger=compound_trigger,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，通过 classify_intent 工具返回结果。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。
意图边界：
- request 表示用户要求修改、取消或更新某项内容，例如修改收货地址、取消订单或变更订单信息；
- logistics 表示用户询问物流状态、配送时效或运输节点。只有询问物流状态、配送时效、运输进度时才返回 logistics；
- 如果用户是在要求执行地址、订单或其他信息变更，即使对象涉及收货地址，也优先返回 request，不要仅因出现“地址”或“订单”就判断为 logistics。

        {ctx}
        用户消息: "{message}"

请务必调用 classify_intent 工具。intent 必须是给定枚举中的一个值，confidence 必须是 0 到 1 之间的数字。

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                tools=[_INTENT_TOOL],
                tool_choice={"type": "tool", "name": _INTENT_TOOL_NAME},
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            data = self._extract_intent_tool_input(resp.content)
            if data is None:
                raise ValueError(f"未收到有效的 {_INTENT_TOOL_NAME} 工具调用")

            intent = data.get("intent")
            if not isinstance(intent, str):
                raise ValueError("工具返回的 intent 不是字符串")
            intent = _INTENT_ALIASES.get(intent, intent)
            try:
                intent = IntentCategory(intent)
            except ValueError as ex:
                raise ValueError(f"工具返回了未注册的 intent: {intent!r}") from ex

            confidence = data.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("工具返回的 confidence 不是数字")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"工具返回的 confidence 超出范围: {confidence}")

            reasoning = data.get("reasoning", "")
            if not isinstance(reasoning, str):
                raise ValueError("工具返回的 reasoning 不是字符串")
            return {
                "intent": intent,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    @staticmethod
    def _extract_intent_tool_input(content: Any) -> Optional[Dict[str, Any]]:
        """从 Anthropic 风格响应中提取指定工具调用的参数。"""
        for block in content or []:
            block_type = getattr(block, "type", None)
            name = getattr(block, "name", None)
            input_data = getattr(block, "input", None)
            if isinstance(block, dict):
                block_type = block.get("type", block_type)
                name = block.get("name", name)
                input_data = block.get("input", input_data)
            if (
                block_type == "tool_use"
                and name == _INTENT_TOOL_NAME
                and isinstance(input_data, dict)
            ):
                return input_data
        return None

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _detect_secondary_intents(
        self,
        message: str,
        primary_intent: IntentCategory,
    ) -> tuple[List[IntentCategory], str]:
        """只为明确的跨域复合请求补充次意图。

        这是一个独立于三路主意图投票的保守规则层：先要求出现明确的复合
        连接词，再把分句分别映射到已有意图枚举。单纯关键词共现不会触发，
        因此不会把“退款页面打不开”硬拆为退款和技术两个意图。
        """
        text = self._clean_text(message).strip()
        if primary_intent == IntentCategory.HUMAN_HANDOFF:
            # 人工升级是终止路由；业务背景保留在原始消息/交接摘要中，
            # 不再将其建模为第二个业务意图或辅助 Agent。
            return [], ""
        trigger = _COMPOUND_STRONG_TRIGGER_RE.search(text)
        if not text or trigger is None:
            return [], ""

        clauses = [
            clause.strip()
            for clause in _COMPOUND_CLAUSE_SPLIT_RE.split(text)
            if clause and clause.strip()
        ]
        if len(clauses) < 2 and primary_intent != IntentCategory.HUMAN_HANDOFF:
            return [], ""

        clause_intents: List[IntentCategory] = []
        for clause in clauses:
            candidates: List[tuple[int, IntentCategory]] = []
            lowered_clause = clause.casefold()
            for intent, keywords in _COMPOUND_PATTERNS.items():
                matched_lengths = [
                    len(keyword)
                    for keyword in keywords
                    if keyword.casefold() in lowered_clause
                ]
                if matched_lengths:
                    candidates.append((max(matched_lengths), intent))
            if candidates:
                # 一个普通分句只保留最明确的业务候选，避免在一个诉求内制造
                # 多个意图；但“要求人工 + 业务背景”是明确的双层语义，必须
                # 同时保留人工请求和业务背景，否则会丢掉人工主意图的上下文。
                candidates.sort(key=lambda item: (-item[0], item[1].value))
                clause_intents.append(candidates[0][1])
                if candidates[0][1] == IntentCategory.HUMAN_HANDOFF:
                    business_candidate = next(
                        (
                            intent
                            for _, intent in candidates
                            if intent != IntentCategory.HUMAN_HANDOFF
                        ),
                        None,
                    )
                    if business_candidate is not None:
                        clause_intents.append(business_candidate)

        candidate_intents = list(dict.fromkeys(clause_intents))
        # 细粒度意图已经代表同一业务域时，去掉同域的泛化标签，
        # 例如“申请退款”同时命中 refund 和 billing，只保留 refund。
        specific_domains = {
            _intent_domain(intent)
            for intent in candidate_intents
            if intent in _SPECIFIC_INTENTS and intent != IntentCategory.HUMAN_HANDOFF
        }
        candidate_intents = [
            intent
            for intent in candidate_intents
            if not (
                intent in _GENERIC_INTENTS
                and _intent_domain(intent) in specific_domains
            )
        ]
        if len(candidate_intents) < 2:
            return [], ""

        primary_domain = _intent_domain(primary_intent)
        secondary: List[IntentCategory] = []
        for intent in candidate_intents:
            if intent == primary_intent:
                continue
            # 人工请求是路由主意图，不作为普通业务的次意图；反过来，
            # 人工主意图需要保留订单/退款/技术等原始业务背景。
            if primary_intent != IntentCategory.HUMAN_HANDOFF and intent == IntentCategory.HUMAN_HANDOFF:
                continue
            if primary_intent != IntentCategory.HUMAN_HANDOFF and _intent_domain(intent) == primary_domain:
                continue
            secondary.append(intent)

        if not secondary:
            return [], ""
        return secondary[:3], trigger.group(0)

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工", "找经理", "supervisor"],
            IntentCategory.ORDER_STATUS: ["订单状态", "发货了吗", "处理到哪", "order status"],
            IntentCategory.LOGISTICS: ["物流", "快递", "配送", "运单", "delivery", "shipping"],
            IntentCategory.REFUND: ["退款", "退货", "refund", "return"],
            IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
            IntentCategory.PAYMENT_ISSUE: ["重复扣款", "多扣", "支付失败", "扣费", "payment failed"],
            IntentCategory.ACCOUNT_SECURITY: ["被盗", "异常登录", "重置密码", "两步验证", "安全"],
            IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "401", "验证码"],
            IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "500", "报错", "crash"],
            IntentCategory.TECHNICAL: [
                "页面打不开", "页面无法打开", "页面加载", "加载不出来",
                "网络请求失败", "网络失败", "访问页面", "403",
            ],
        }
        generic_patterns = {
            IntentCategory.COMPLAINT:  ["投诉", "太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            pattern_intent = pat.get("intent", IntentCategory.OTHER)
            pattern_confidence = float(pat.get("confidence", 0.0) or 0.0)
            if pattern_intent in _SPECIFIC_INTENTS and pattern_confidence > 0:
                return pattern_intent, source_scores["pattern"], source_scores

            embedding_confidence = float(emb.get("confidence", 0.0) or 0.0)
            if (
                emb.get("intent") != IntentCategory.OTHER
                and embedding_confidence >= self.embedding_fallback_threshold
            ):
                return emb["intent"], source_scores["embedding"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best]
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS and pat_conf >= 0.5 and best_score < 0.8:
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, re.I)
                + re.findall(r"\b([45]\d{2})\b", message)
            ),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.HUMAN_HANDOFF:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _intent_value(value: Any) -> str:
        if isinstance(value, IntentCategory):
            return value.value
        if isinstance(value, str):
            value = _INTENT_ALIASES.get(value, value)
        try:
            return IntentCategory(value).value
        except (TypeError, ValueError):
            return IntentCategory.OTHER.value

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
