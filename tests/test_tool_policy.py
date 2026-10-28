"""验证工具白名单、只读次数和删除工具的审批配置；不调用模型。"""

from unittest.mock import Mock

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage

from agent_demo.tool_policy import TOOL_POLICIES, build_tool_guard, build_tool_middleware
from agent_demo.tools import build_tools


def request(name: str, call_id: str = "call-1") -> ToolCallRequest:
    """造一个只带工具名和调用 ID 的请求，供权限测试使用。"""
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id},
        tool=None, state={}, runtime=None,
    )


def ok(req):
    return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])


def test_guard_allows_known_tool():
    guard = build_tool_guard()
    handler = Mock(side_effect=ok)
    assert guard.wrap_tool_call(request("calculator"), handler).content == "ok"
    handler.assert_called_once()


def test_guard_denies_unknown_tool():
    guard = build_tool_guard()
    handler = Mock()
    result = guard.wrap_tool_call(request("delete_everything"), handler)
    assert result.status == "error" and "未被授权" in result.content
    handler.assert_not_called()


def test_guard_limits_read_only_tool():
    guard = build_tool_guard()
    handler = Mock(side_effect=ok)
    limit = TOOL_POLICIES["current_time"]["limit"]
    for index in range(limit):
        assert guard.wrap_tool_call(request("current_time", f"c{index}"), handler).content == "ok"
    denied = guard.wrap_tool_call(request("current_time", "extra"), handler)
    assert denied.status == "error" and "上限" in denied.content
    assert handler.call_count == limit


def test_delete_tool_has_no_limit():
    guard = build_tool_guard()
    handler = Mock(side_effect=ok)
    for index in range(20):
        assert guard.wrap_tool_call(request("delete_knowledge", f"d{index}"), handler).content == "ok"
    assert handler.call_count == 20


def test_policy_covers_every_registered_tool():
    assert set(TOOL_POLICIES) == {tool.name for tool in build_tools(Mock())}


def test_only_delete_requires_approval():
    middleware = build_tool_middleware()
    assert len(middleware) == 2
    assert set(middleware[1].interrupt_on) == {"delete_knowledge"}
