"""直接加总模型返回的数字：本轮用量 + 上轮累计 = 新累计。"""

from typing import Any

from langchain_core.messages import AIMessage

# 五项统计共用同一套“完整才相加”的规则，集中定义避免两处写漏。
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")


def summarize_usage(
    messages: list[AIMessage | None], previous: dict[str, Any], chat_seconds: float,
) -> dict[str, Any]:
    """参数：本轮模型消息、上轮累计、整轮耗时；返回含 current 与 total 的字典。"""
    records = []
    for message in messages:
        # 请求失败时调用方传入 None；接口没给的数字保留 None，不能猜成 0。
        usage = (message.usage_metadata or {}) if isinstance(message, AIMessage) else {}
        records.append({
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": (usage.get("input_token_details") or {}).get("cache_read"),
            "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
        })

    current = {}
    total = {}
    for field in USAGE_FIELDS:
        values = [record[field] for record in records]
        # 缺一笔就不算完整合计；缓存和推理已包含在输入、输出里，不重复加到总 token。
        current[field] = sum(values) if values and all(value is not None for value in values) else None
        before = previous.get(field, 0)  # 新会话没有上轮，累计从 0 开始。
        total[field] = before + current[field] if before is not None and current[field] is not None else None

    current["chat_seconds"] = chat_seconds
    total["chat_seconds"] = previous.get("chat_seconds", 0) + chat_seconds
    total["turns"] = previous.get("turns", 0) + 1
    return {"current": current, "total": total}
