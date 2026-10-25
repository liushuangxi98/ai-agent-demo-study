"""直接读取返回值统计用量；真实执行 LangChain，HTTP 返回使用本地虚构数据。"""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

import agent_demo.agent as agent_module
from agent_demo.agent import AgentApp
from agent_demo.usage import summarize_usage


def test_usage_adds_returned_numbers_and_last_snapshot_without_averaging_rates():
    messages = [AIMessage(content="回答", usage_metadata={
        "input_tokens": 900, "output_tokens": 90, "total_tokens": 990,
        "input_token_details": {"cache_read": 0}, "output_token_details": {"reasoning": 0},
    })]
    previous = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "cached_tokens": 100, "reasoning_tokens": 10, "chat_seconds": 2, "turns": 1}
    result = summarize_usage(messages, previous, 3)
    total = result["total"]
    assert total["total_tokens"] == 1100 and total["chat_seconds"] == 5 and total["turns"] == 2
    assert total["cached_tokens"] / total["input_tokens"] == 0.1
    assert total["reasoning_tokens"] / total["output_tokens"] == 0.1
    assert previous["total_tokens"] == 110  # 不原地修改上轮记录。
    assert summarize_usage(messages, previous, 3) == result  # 重读不会重复加。


def test_missing_fields_and_failed_requests_are_not_counted_as_zero():
    message = AIMessage(content="回答", usage_metadata={
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
    })
    result = summarize_usage([message], {}, 2)
    assert result["current"]["total_tokens"] == 12
    assert result["current"]["cached_tokens"] is None
    assert result["current"]["reasoning_tokens"] is None
    assert summarize_usage([message, None], {}, 3)["current"]["total_tokens"] is None
    # 旧累计不完整时，后面也不能把不完整总数冒充完整累计。
    assert summarize_usage([message], {"total_tokens": None}, 1)["total"]["total_tokens"] is None


@pytest.mark.parametrize("reflection_enabled", [False, True])
@pytest.mark.parametrize("bad_memory_output", [False, True])
@pytest.mark.parametrize("with_summary", [False, True])  # 用真实 LangChain 验证摘要请求也进入同一统计链。
def test_direct_results_include_tools_reflection_memory_and_chat_elapsed(
    monkeypatch, settings, reflection_enabled, bad_memory_output, with_summary,  # 同时覆盖摘要、反思与解析失败的组合。
):
    """真实执行 Agent 和结构化解析，只替换最底层 HTTP，不接触真实服务。"""
    llm = ChatOpenAI(model="test-model", api_key="local-test-placeholder", use_responses_api=False)
    responses = []

    def reply(input_tokens, output_tokens, cached_tokens, reasoning_tokens, content=None, tool=None):
        message = {"role": "assistant", "content": content}
        if tool:
            name, arguments = tool
            message["tool_calls"] = [{"id": "call-" + str(len(responses)), "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(arguments)}}]
        raw = Mock(headers={})
        raw.parse.return_value = {
            "id": "fake-response", "model": "test-model", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                      "total_tokens": input_tokens + output_tokens,
                      "prompt_tokens_details": {"cached_tokens": cached_tokens},
                      "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
        }
        responses.append(raw)

    reply(1577, 31, 0, 15, tool=("calculator", {"expression": "1+2"}))
    reply(1635, 104, 1024, 42, content="结果是50")
    if reflection_enabled:
        reply(100, 10, 50, 0, tool=("ReviewResult", {"issues": [{
            "kind": "tool", "quote": "50", "reason": "计算器结果为3", "suggestion": "改为3",
        }]}))
        reply(200, 20, 100, 0, content="结果是3")
    reply(80, 8, 0, 0, tool=("MemoryPatch", {"changes": "不合格的列表" if bad_memory_output else []}))
    if with_summary:  # 摘要发生在主回答、反思和画像提取结束之后。
        reply(200, 20, 0, 0, content="旧问题与旧回答的累计摘要")  # 真实客户端解析普通摘要消息。
    request = Mock(side_effect=responses)
    monkeypatch.setattr(llm.client.with_raw_response, "create", request)

    app = AgentApp.__new__(AgentApp)
    app.settings = replace(settings, reflection_enabled=reflection_enabled, context_token_budget=1 if with_summary else 50000, summary_keep_turns=1)  # 使用低阈值制造摘要场景。
    app.llm = llm
    app.memory = Mock()
    # 故意给历史消息附带很大的旧用量，确认它不会再被计算到本轮。
    app.memory.context.return_value = (  # 历史行携带旧统计，但主流程只把问答正文转换成消息。
        [{"id": 1, "user_text": "之前的问题", "assistant_text": "之前的回答", "usage_json": {"current": {"total_tokens": 100000}}}],  # 旧轮用量不能重复计算。
        {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cached_tokens": 50, "reasoning_tokens": 5, "chat_seconds": 2, "turns": 1},  # 上轮累计。
        {},  # 尚未产生摘要。
    )  # 三项上下文返回值结束。
    app.memory.profile.return_value = {}
    app.knowledge = Mock()
    app.knowledge.context.return_value = "知识库未找到足够相关的证据。"

    clock = iter([10.0, 17.5])
    monkeypatch.setattr(agent_module, "perf_counter", lambda: next(clock))
    result = app.chat("usage-study", "计算1+2")
    stats = result["usage"]["current"]
    assert stats["input_tokens"] == 3292 + (300 if reflection_enabled else 0) + (200 if with_summary else 0)  # 摘要输入另计但不改变触发信号。
    assert stats["output_tokens"] == 143 + (30 if reflection_enabled else 0) + (20 if with_summary else 0)  # 摘要输出同样计入。
    assert stats["total_tokens"] == 3435 + (330 if reflection_enabled else 0) + (220 if with_summary else 0)  # 所有请求各加一次。
    assert stats["cached_tokens"] == 1024 + (150 if reflection_enabled else 0)
    assert stats["reasoning_tokens"] == 57
    assert request.call_count == (5 if reflection_enabled else 3) + int(with_summary)  # 摘要最多增加一次模型调用。
    assert stats["chat_seconds"] == 7.5
    total = result["usage"]["total"]
    assert total == {
        "input_tokens": stats["input_tokens"] + 100, "output_tokens": stats["output_tokens"] + 10,
        "total_tokens": stats["total_tokens"] + 110, "cached_tokens": stats["cached_tokens"] + 50,
        "reasoning_tokens": stats["reasoning_tokens"] + 5, "chat_seconds": 9.5, "turns": 2,
    }
    assert bool(result["memory_warning"]) == bad_memory_output
    # 解析失败不等于接口没返回：提取记忆这次的 80+8 token 仍然计入。
    saved_trace = app.memory.save_turn.call_args.args[3]
    assert all(item.get("kind") != "usage" for item in saved_trace)
    assert app.memory.save_turn.call_args.kwargs["usage"] == result["usage"]
    if with_summary:  # 保存路径必须接收真实摘要与原文边界。
        assert app.memory.save_turn.call_args.kwargs["summary"] == {"text": "旧问题与旧回答的累计摘要", "through_turn_id": 1}  # 不能用消息条数替代数据库 ID。
        summary_request = request.call_args.kwargs  # 最后一次 HTTP 请求属于摘要。
        assert not any(key in summary_request for key in ("max_tokens", "max_completion_tokens", "tools"))  # 摘要不限应用层长度，也没有业务工具。
    # 用量走独立字段，不作为“工具证据”发回模型。
    if reflection_enabled:
        review_payload = request.call_args_list[2].kwargs["messages"][1]["content"]
        assert not any(item.get("kind") == "usage" for item in json.loads(review_payload)["materials"]["tool_results"])
