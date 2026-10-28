# tool_policy.py：工具权限与人工审批

源码：[agent_demo/tool_policy.py](D:/study/AI/ai-agent-demo-study/agent_demo/tool_policy.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

`tool_policy.py` 是**工具房间门口的门卫**：模型递来工具单子时，先看这个工具允不允许、本轮调用几次；需要人点头的，再交给 `HumanInTheLoopMiddleware` 暂停等审批。

它不执行工具，也不碰数据库。真正执行在 `tools.py`，审批恢复和保存由 `agent.py` 串起来。

## 1. 策略表：一个表管三件事

```python
TOOL_POLICIES = {
    "calculator":       {"limit": 10, "approval": False},
    "current_time":     {"limit": 3,  "approval": False},
    "knowledge_search": {"limit": 5,  "approval": False},
    "delete_knowledge": {"limit": None, "approval": True},
}
```

| 字段 | 白话 |
| --- | --- |
| 工具名（键） | 白名单；不在表里的工具一律拒绝 |
| `limit` | 每轮最多调用几次；`None` 表示不限 |
| `approval` | 是否必须在执行前等人工批准 |

删除工具**不限次数**，因为审批本身就是更强的控制；只读工具有次数上限，防止模型陷入循环。策略写在代码里，**没有 `config.py` / `.env` 开关**，永远生效。

## 2. 两个中间件怎么分工

| 中间件 | 何时跑 | 干什么 |
| --- | --- | --- |
| `tool_guard`（自定义） | 每个工具执行前，`wrap_tool_call` | 查白名单、只读工具计数；拒绝时返回 `ToolMessage(status="error")` |
| `HumanInTheLoopMiddleware`（内置） | 模型返回后、工具执行前 | 对 `approval=True` 的工具 `interrupt()`，暂停等批准/拒绝 |

逻辑顺序是**白名单 → 次数 → 审批 → 执行**。实现上 HITL 在 `after_model` 暂停，权限门卫在恢复后的 `wrap_tool_call` 再检查；因为 `interrupt_on` 只包含策略表里已登记的工具，实际不会出现“先问人、批完才发现未授权”。删除工具没有次数限制，所以也不存在“拒绝后次数被扣”的问题。

**不需要自己写 `before_model`。** `before_model` 在模型回答前跑，那时还看不到工具调用；权限用 `wrap_tool_call`，审批由 HITL 内部用 `after_model` + `interrupt()` + `wrap_tool_call` 完成。

## 3. `build_tool_guard()`

```python
@wrap_tool_call
def guard(request, handler):
    name = request.tool_call["name"]
    policy = TOOL_POLICIES.get(name)
    if policy is None:
        return ToolMessage(content=f"工具 {name} 未被授权使用。", ...)
    limit = policy["limit"]
    if limit is not None:
        used = counts.get(name, 0)
        if used >= limit:
            return ToolMessage(content=f"工具 {name} 本轮调用次数已达上限（{limit} 次）。", ...)
        counts[name] = used + 1
    return handler(request)
```

- 计数器是闭包里的字典；`chat()` 每轮建一张图、一份计数，`resume()` 复用同一张图，所以同一轮里计数跨审批暂停保留，下一轮才清零。
- 拒绝返回 ToolMessage，不抛异常；模型能看到原因，换工具或直接回答。
- 它不处理审批；审批由下面的 HITL 负责。

## 4. `build_tool_middleware()`

```python
middleware = [build_tool_guard()]
interrupt_on = {
    name: {"allowed_decisions": ["approve", "reject"]}
    for name, policy in TOOL_POLICIES.items()
    if policy["approval"]
}
middleware.append(HumanInTheLoopMiddleware(interrupt_on, description_prefix="以下工具调用需要人工批准"))
return middleware
```

`interrupt_on` 从策略表生成，只有 `delete_knowledge` 会暂停。没列出的工具自动放行。

## 5. 暂停和恢复：checkpointer 是什么

审批暂停时，图里有历史消息、模型刚说出的工具调用、参数等状态。`checkpointer` 就是**存档**：按 `thread_id`（这里用 `session_id`）保存，批准后从同一个存档继续。

```python
self.checkpointer = InMemorySaver()
result = agent.invoke(
    {"messages": input_messages},
    config={"recursion_limit": 12, "configurable": {"thread_id": session_id}},
)
if result.get("__interrupt__"):
    return {"status": "pending_approval", "actions": ...}
```

批准后：

```python
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config={"configurable": {"thread_id": session_id}},
)
```

拒绝就用 `{"type": "reject", "message": "用户拒绝"}`。

- `InMemorySaver` 是内存存档，程序重启就丢；本地教学够用。企业级需要持久化 checkpointer。
- 它和 MySQL 聊天记录不是一回事：MySQL 存长期聊天，checkpointer 存“这轮暂停到一半的现场”。

## 6. 一轮审批实际发生什么

你说：“删除知识库里的 course.md。”

1. 模型请求 `delete_knowledge(source="course.md")`；
2. `tool_guard` 查白名单：允许；删除工具不限次数；
3. HITL 发现需要审批，`interrupt()`，图暂停；文件仍在，索引未动；
4. `app.chat()` 返回 `status="pending_approval"`，**不保存、不提取画像、不摘要**；
5. 网页显示审批卡片：工具名、参数、说明，批准/拒绝；
6. 点“批准”后 `app.resume()` 用同一个 `thread_id` 恢复；
7. 工具执行：删原件 → `ingest()` 清旧切片；
8. 模型第二次被调用，写最终回答；
9. 保存问答、轨迹（含 `kind="approval"`）、用量和画像。

点“拒绝”：工具不执行，模型收到拒绝的 ToolMessage，正常写回答并保存。

**是一个用户回合、两次模型调用**：第一次模型提出工具调用，第二次模型消化工具结果。审批只是在中间插入人工暂停，不是两次独立对话。

## 7. 失败和边界

| 情况 | 行为 |
| --- | --- |
| 未知工具 | 权限门卫拒绝，不执行 |
| 只读工具超次数 | 返回“已达上限”，模型可继续回答 |
| 删除工具被拒绝 | 不删除；模型收到拒绝，本轮正常保存 |
| 待审批时又问新问题 | `chat()` 报“请先批准或拒绝” |
| 程序重启 | 内存 checkpointer 与 pending 丢失，待审批作废 |
| 多个待审批动作 | 最小版“批准全部/拒绝全部” |
| 删除后知识库为空 | `ingest()` 清空索引，下一轮 `context()` 返回“知识库为空” |
| 评测脚本遇到待审批 | 无人在场，自动拒绝并继续评测 |

**不是所有工具都审批**：只读工具自动放行；给查询也弹窗会训练用户闭眼点“批准”，审批就失去意义。
