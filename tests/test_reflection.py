"""不联网检查反思流程：能纠错、不过度改写、失败时保留草稿并说清状态。"""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

import agent_demo.agent as agent_module
import agent_demo.config as config_module
import agent_demo.reflection as reflection_module
from agent_demo.agent import AgentApp
from agent_demo.reflection import ReviewIssue, ReviewResult, reflect_answer


@pytest.fixture
def materials():
    """只用虚构的题目和工具结果，不碰个人聊天或知识库。"""
    return {
        "question": "计算 (12+8)*3，并说明用了什么工具。", "history": [], "profile": "{}",
        "tool_results": [{"tool": "calculator", "args": {"expression": "(12+8)*3"}, "result": "60"}],
        "answer_rules": "计算用 calculator；引用必须来自本轮资料。",
    }


def wrong_number():
    """准备一张有效的检查单，后面几种修订分支共用它。"""
    return ReviewResult(issues=[ReviewIssue(
        kind="tool", quote="结果是50", reason="calculator 返回60", suggestion="把50改为60",
    )])


def test_correct_draft_is_kept_without_rewriting(materials):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": ReviewResult(issues=[]), "parsing_error": None,
    }

    answer, note, messages = reflect_answer(llm, materials, "结果是60，使用了计算器。")

    assert answer == note["draft"] == "结果是60，使用了计算器。"
    assert note["status"] == "passed"
    llm.with_structured_output.return_value.invoke.assert_called_once()
    llm.invoke.assert_not_called()  # 没发现问题，不应白花一次重写请求。


def test_wrong_draft_is_rewritten_only_once(materials):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": wrong_number(), "parsing_error": None,
    }
    llm.invoke.return_value = AIMessage(content="结果是60，使用了计算器。")

    answer, note, messages = reflect_answer(llm, materials, "结果是50")

    assert answer == "结果是60，使用了计算器。"
    assert note["draft"] == "结果是50"
    assert note["status"] == "revised"
    assert note["issues"][0]["quote"] == "结果是50"
    llm.with_structured_output.assert_called_once()
    llm.invoke.assert_called_once()
    assert "未进行第二次" in note["result"]


@pytest.mark.parametrize("kind,quote,valid", [
    ("tool", "草稿里没有这句", False),
    ("missing", "并说明用了什么工具", True),
    ("missing", "用户没有问过这句", False),
])
def test_review_must_locate_the_original_sentence(materials, kind, quote, valid):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsing_error": None,
        "parsed": ReviewResult(issues=[ReviewIssue(
            kind=kind, quote=quote, reason="测试依据", suggestion="测试修改建议",
        )]),
    }
    llm.invoke.return_value = AIMessage(content="结果是60，使用计算器。")
    _, note, _ = reflect_answer(llm, materials, "结果是60")
    assert note["status"] == ("revised" if valid else "review_failed")


@pytest.mark.parametrize("reply", [{"issues": "不是列表"}, {"issues": [], "invented": True}])
def test_bad_review_format_preserves_draft(materials, reply):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": reply, "parsing_error": None,
    }
    answer, note, messages = reflect_answer(llm, materials, "草稿")
    assert answer == "草稿" and note["status"] == "review_failed"
    llm.invoke.assert_not_called()


def test_review_service_failure_is_visible_without_leaking_details(materials):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("测试秘密不能写进记录")
    answer, note, messages = reflect_answer(llm, materials, "草稿")
    assert answer == "草稿" and note["status"] == "review_failed"
    assert "RuntimeError" in note["result"]
    assert "测试秘密" not in json.dumps(note, ensure_ascii=False)
    llm.invoke.assert_not_called()


@pytest.mark.parametrize("reply", [
    AIMessage(content="  "),
    AIMessage(content="", tool_calls=[{"id": "c1", "name": "calculator", "args": {"expression": "1+1"}}]),
    AIMessage(content="结果是60。[invented.md#0]"),
    RuntimeError("测试秘密不能写进记录"),
])
def test_failed_or_invalid_revision_retains_issues_and_draft(materials, reply):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": wrong_number(), "parsing_error": None,
    }
    if isinstance(reply, Exception):
        llm.invoke.side_effect = reply
    else:
        llm.invoke.return_value = reply

    answer, note, messages = reflect_answer(llm, materials, "结果是50")

    assert answer == "结果是50" and note["status"] == "revision_failed"
    assert note["issues"] and "未完成修正" in note["result"]
    assert "测试秘密" not in json.dumps(note, ensure_ascii=False)
    llm.invoke.assert_called_once()


def test_program_catches_fake_citation_and_allows_its_removal(materials):
    llm = Mock()
    # 故意让模型漏看假引用，验证 Python 会补上一条问题。
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": ReviewResult(issues=[]), "parsing_error": None,
    }
    llm.invoke.return_value = AIMessage(content="现有资料没有提供授课老师的姓名。")

    answer, note, messages = reflect_answer(llm, materials, "授课老师叫小甲。[invented.md#0]")

    assert note["status"] == "revised"
    assert note["issues"][0]["kind"] == "citation"
    assert "[invented.md#0]" not in answer  # 修正可以删引用，不是要求原样保留。


def test_citations_from_later_search_are_also_valid(materials):
    materials["tool_results"].extend([
        {"tool": "knowledge_search", "result": "[first.md#0]\n第一份资料。"},
        {"tool": "knowledge_search", "result": "[later.md#2]\n补充资料。"},
    ])
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": ReviewResult(issues=[]), "parsing_error": None,
    }
    _, note, _ = reflect_answer(llm, materials, "补充资料。[later.md#2]")
    assert note["status"] == "passed"
    llm.invoke.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_chat_checks_complete_materials_and_saves_only_selected_answer(monkeypatch, enabled, settings):
    # 只替换外部服务，执行真实 chat()、检查、重写和保存参数组装。
    app = AgentApp.__new__(AgentApp)
    app.settings = replace(settings, reflection_enabled=enabled)
    app.llm = Mock()
    app.pending = {}
    app.checkpointer = Mock()
    app.llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": wrong_number(), "parsing_error": None,
    }
    app.llm.invoke.return_value = AIMessage(content="结果是60")
    app.memory = Mock()
    app.memory.context.return_value = ([{"id": 1, "user_text": "之前的练习", "assistant_text": "继续"}], {}, {})  # 反思读取未压缩原文和单独摘要。
    app.memory.profile.return_value = {
        "旧计划": {"value": "不能当现状的旧计划", "forgotten": False, "expires_on": "2000-01-01"},
        "旧偏好": {"value": "不能恢复的旧偏好", "forgotten": True, "expires_on": None},
    }
    app.knowledge = Mock()
    app.knowledge.context.return_value = "[first.md#0]\n首次检索。"
    graph = Mock()
    graph.invoke.side_effect = lambda state, config: {"messages": state["messages"] + [
        AIMessage(content="", tool_calls=[
            {"id": "s1", "name": "knowledge_search", "args": {"query": "补查"}},
            {"id": "c1", "name": "calculator", "args": {"expression": "(12+8)*3"}},
        ]),
        ToolMessage(content="[later.md#2]\n补充资料。", tool_call_id="s1"),
        ToolMessage(content="60", tool_call_id="c1"), AIMessage(content="结果是50"),
    ]}
    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=graph))
    extractor = Mock(return_value=([], None, ""))
    monkeypatch.setattr(agent_module, "extract_changes", extractor)
    reflector = Mock(wraps=reflection_module.reflect_answer)
    monkeypatch.setattr(reflection_module, "reflect_answer", reflector)

    question = "计算 (12+8)*3"
    result = app.chat("study-01", question)

    assert result["answer"] == ("结果是60" if enabled else "结果是50")
    assert app.memory.save_turn.call_args.args[1:3] == (question, result["answer"])
    assert app.memory.save_turn.call_args.args[3] == result["trace"]
    assert extractor.call_args.args[1] == question  # 检查意见不参与用户事实提取。
    graph.invoke.assert_called_once()  # 修订没有重跑 Agent 或工具。
    if enabled:
        reflector.assert_called_once()
        sent = json.loads(app.llm.with_structured_output.return_value.invoke.call_args.args[0][1].content)
        supplied = sent["materials"]
        assert supplied["question"] == question and supplied["history"][0]["text"] == "之前的练习"
        assert "不能当现状" not in supplied["profile"] and "不能恢复" not in supplied["profile"]
        assert "已过期" in supplied["profile"] and "已遗忘" in supplied["profile"]
        assert [item["tool"] for item in supplied["tool_results"]] == [
            "knowledge_search", "knowledge_search", "calculator",
        ]
        assert result["trace"][-1]["status"] == "revised"
    else:
        reflector.assert_not_called()
        app.llm.with_structured_output.assert_not_called()
        app.llm.invoke.assert_not_called()


@pytest.mark.parametrize("value,expected", [(None, False), ("false", False), ("true", True), (" YES ", True)])
def test_environment_switch_does_not_treat_false_as_true(monkeypatch, value, expected, settings):
    # 用虚构的环境变量测试读取和自动检查，不读取实际 .env。
    monkeypatch.setattr(config_module, "load_dotenv", Mock())
    env = {
        "LLM_API_KEY": settings.llm_api_key,
        "LLM_BASE_URL": settings.llm_base_url,
        "MYSQL_USER": settings.mysql_user,
        "MYSQL_PASSWORD": settings.mysql_password,
    }
    if value is not None:
        env["REFLECTION_ENABLED"] = value
    monkeypatch.setattr(config_module, "os", SimpleNamespace(environ=env))
    assert config_module.load_settings().reflection_enabled is expected


def test_real_langchain_parser_understands_review_function_call(monkeypatch, materials):
    # 只替换最末端 HTTP，结构化输出的打包和解析仍走真实 LangChain。
    llm = ChatOpenAI(model="test-model", api_key="local-test-placeholder", use_responses_api=False)
    response = {
        "id": "review-test", "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{
                "id": "review-call", "type": "function", "function": {
                    "name": "ReviewResult", "arguments": wrong_number().model_dump_json(),
                },
            }],
        }}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    raw = Mock(headers={})
    raw.parse.return_value = response
    revised_raw = Mock(headers={})
    revised_raw.parse.return_value = {
        **response,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "结果是60"}}],
    }
    request = Mock(side_effect=[raw, revised_raw])
    monkeypatch.setattr(llm.client.with_raw_response, "create", request)

    _, note, messages = reflect_answer(llm, materials, "结果是50")
    assert note["issues"] == [issue.model_dump() for issue in wrong_number().issues]
    assert messages[0].usage_metadata["total_tokens"] == 2
    assert request.call_args_list[0].kwargs["tool_choice"]["function"]["name"] == "ReviewResult"
    assert request.call_args_list[0].kwargs["extra_body"] == {"enable_thinking": False}


def test_invalid_review_and_revision_still_return_original_usage(materials):
    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    raw = AIMessage(content="", usage_metadata=usage)
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": raw, "parsed": None, "parsing_error": ValueError("解析失败"),
    }
    _, note, messages = reflect_answer(llm, materials, "结果是50")
    assert note["status"] == "review_failed" and messages == [raw]
    assert messages[0].usage_metadata["total_tokens"] == 110

    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": raw, "parsed": wrong_number(), "parsing_error": None,
    }
    revised = AIMessage(content="", usage_metadata=usage)  # 接口有返回，但修订稿无效。
    llm.invoke.return_value = revised
    _, note, messages = reflect_answer(llm, materials, "结果是50")
    assert note["status"] == "revision_failed" and messages == [raw, revised]
