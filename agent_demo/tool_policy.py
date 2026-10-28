"""工具权限：声明式白名单、只读工具次数上限、删除工具人工审批。"""

from langchain.agents.middleware import HumanInTheLoopMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage

# 工具名 -> 每轮最多调用次数（None 不限） + 是否需要人工审批。
# 不在表里的工具一律拒绝；新增工具必须在这里登记。
TOOL_POLICIES = {
    "calculator": {"limit": 10, "approval": False},
    "current_time": {"limit": 3, "approval": False},
    "knowledge_search": {"limit": 5, "approval": False},
    "delete_knowledge": {"limit": None, "approval": True},
}


def build_tool_guard():
    """返回权限门卫；计数随中间件创建，一次 invoke 一份。"""
    counts: dict[str, int] = {}

    @wrap_tool_call
    def guard(request, handler):
        name = request.tool_call["name"]
        policy = TOOL_POLICIES.get(name)
        if policy is None:
            return ToolMessage(
                content=f"工具 {name} 未被授权使用。",
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        limit = policy["limit"]
        if limit is not None:
            used = counts.get(name, 0)
            if used >= limit:
                return ToolMessage(
                    content=f"工具 {name} 本轮调用次数已达上限（{limit} 次）。",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            counts[name] = used + 1
        return handler(request)

    return guard


def build_tool_middleware():
    """返回本轮中间件：权限门卫 + 需要审批的工具。"""
    middleware = [build_tool_guard()]
    interrupt_on = {
        name: {"allowed_decisions": ["approve", "reject"]}
        for name, policy in TOOL_POLICIES.items()
        if policy["approval"]
    }
    if interrupt_on:
        middleware.append(
            HumanInTheLoopMiddleware(interrupt_on, description_prefix="以下工具调用需要人工批准")
        )
    return middleware
