"""验证“待审批 → 恢复”的两段式流程；假图 + 假 HTTP，不调用真实模型。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent_demo.agent as agent_module
from agent_demo.agent import AgentApp
from agent_demo.tool_policy import build_tool_middleware
from agent_demo.tools import build_tools


def reply(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata={
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
        "input_token_details": {"cache_read": 0},
        "output_token_details": {"reasoning": 0},
    })


@pytest.mark.parametrize("decision, status", [
    ({"type": "approve"}, "approved"),
    ({"type": "reject", "message": "用户拒绝"}, "rejected"),
])
def test_chat_pauses_then_resume_saves_once(monkeypatch, settings, decision, status):
    app = AgentApp.__new__(AgentApp)
    app.settings = settings
    app.llm = Mock()
    app.memory = Mock()
    app.memory.context.return_value = ([], {}, {})
    app.memory.profile.return_value = {}
    app.knowledge = Mock()
    app.knowledge.context.return_value = "知识库未找到足够相关的证据。"
    app.pending = {}
    app.checkpointer = Mock()

    captured = []
    tool_call = {"name": "delete_knowledge", "args": {"source": "course.md"},
                 "id": "del-1", "type": "tool_call"}

    def invoke(state, config):
        if isinstance(state, dict):
            captured[:] = state["messages"]
            return {"__interrupt__": [SimpleNamespace(value={"action_requests": [{
                "name": "delete_knowledge", "args": {"source": "course.md"},
                "description": "删除知识文档",
            }]})]}
        return {"messages": captured + [
            AIMessage(content="", tool_calls=[tool_call], usage_metadata={
                "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                "input_token_details": {"cache_read": 0},
                "output_token_details": {"reasoning": 0},
            }),
            ToolMessage(content="已删除" if status == "approved" else "用户拒绝", tool_call_id="del-1"),
            reply("已处理"),
        ]}

    graph = Mock()
    graph.invoke.side_effect = invoke
    builder = Mock(return_value=graph)
    monkeypatch.setattr(agent_module, "create_agent", builder)
    monkeypatch.setattr(agent_module, "extract_changes", Mock(return_value=([], None, "")))

    pending = app.chat("approval-session", "删除 course.md")

    assert pending["status"] == "pending_approval"
    assert pending["actions"][0]["tool"] == "delete_knowledge"
    assert app.pending["approval-session"]
    app.memory.save_turn.assert_not_called()

    result = app.resume("approval-session", decision)

    assert result["answer"] == "已处理"
    assert [item["tool"] for item in result["trace"]] == [
        "knowledge_search", "approval", "delete_knowledge",
    ]
    assert result["trace"][1]["status"] == status
    assert builder.call_count == 1
    app.memory.save_turn.assert_called_once()
    assert app.memory.save_turn.call_args.args[1] == "删除 course.md"
    assert app.pending == {}


def test_chat_refuses_a_second_question_while_pending(monkeypatch, settings):
    app = AgentApp.__new__(AgentApp)
    app.settings = settings
    app.llm = Mock()
    app.memory = Mock()
    app.knowledge = Mock()
    app.pending = {"approval-session": {"actions": [{"tool": "delete_knowledge"}]}}
    app.checkpointer = Mock()

    with pytest.raises(ValueError, match="等待审批"):
        app.chat("approval-session", "另一个问题")


def fake_response(tool=None, content=None):
    """造一份 OpenAI 兼容响应；只替换 HTTP 层，保留真实 LangChain 解析。"""
    response = Mock(headers={})
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{
            "id": "call-del", "type": "function",
            "function": {"name": tool[0], "arguments": json.dumps(tool[1])},
        }]
    response.parse.return_value = {
        "id": "x", "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                  "prompt_tokens_details": {"cached_tokens": 0},
                  "completion_tokens_details": {"reasoning_tokens": 0}},
    }
    return response


def test_real_hitl_interrupts_before_delete_and_resumes_after_approval():
    llm = ChatOpenAI(model="test-model", api_key="local-test-placeholder", use_responses_api=False)
    llm.client.with_raw_response.create = Mock(side_effect=[
        fake_response(("delete_knowledge", {"source": "course.md"})),
        fake_response(content="已删除"),
    ])
    knowledge = Mock()
    knowledge.delete.return_value = "已删除 course.md"
    checkpointer = InMemorySaver()

    def build_agent():
        return create_agent(
            llm, tools=build_tools(knowledge), system_prompt="test",
            middleware=build_tool_middleware(), checkpointer=checkpointer,
        )

    config = {"configurable": {"thread_id": "hitl-test"}}

    first = build_agent().invoke({"messages": [HumanMessage(content="删除 course.md")]}, config=config)

    assert first["__interrupt__"][0].value["action_requests"][0]["name"] == "delete_knowledge"
    knowledge.delete.assert_not_called()

    second = build_agent().invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)

    assert second["messages"][-1].text == "已删除"
    knowledge.delete.assert_called_once_with("course.md")


def test_agent_app_pauses_and_resumes_through_real_hitl(monkeypatch, settings):
    app = AgentApp.__new__(AgentApp)
    app.settings = settings
    app.memory = Mock()
    app.memory.context.return_value = ([], {}, {})
    app.memory.profile.return_value = {}
    app.knowledge = Mock()
    app.knowledge.context.return_value = "知识库未找到足够相关的证据。"
    app.knowledge.delete.return_value = "已删除 course.md"
    app.pending = {}
    app.checkpointer = InMemorySaver()

    llm = ChatOpenAI(model="test-model", api_key="local-test-placeholder", use_responses_api=False)
    llm.client.with_raw_response.create = Mock(side_effect=[
        fake_response(("delete_knowledge", {"source": "course.md"})),
        fake_response(content="已删除"),
    ])
    app.llm = llm
    monkeypatch.setattr(agent_module, "extract_changes", Mock(return_value=([], None, "")))

    pending = app.chat("app-hitl", "删除 course.md")

    assert pending["status"] == "pending_approval"
    app.memory.save_turn.assert_not_called()
    app.knowledge.delete.assert_not_called()

    result = app.resume("app-hitl", {"type": "approve"})

    assert result["answer"] == "已删除"
    assert [item["tool"] for item in result["trace"]] == [
        "knowledge_search", "approval", "delete_knowledge",
    ]
    app.knowledge.delete.assert_called_once_with("course.md")
    app.memory.save_turn.assert_called_once()
    assert app.pending == {}
