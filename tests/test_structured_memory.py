"""验证画像增量、时间、弃用与真实 Agent 接入；所有模型输出使用虚构数据。"""

# ---------------------------------------------------------------------------
# 标准库 / 第三方库导入
# ---------------------------------------------------------------------------
import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

# AIMessage 用来在假图里伪造“模型的回答”。
from langchain_core.messages import AIMessage

# ChatOpenAI 是真模型客户端。这个文件里有一处特意用它来跑真实的解析流程，
# 只把“发 HTTP 请求”那一步替换掉。
from langchain_openai import ChatOpenAI

# ValidationError 是 pydantic 校验失败时抛的异常。
from pydantic import ValidationError

# 下面两个是用来“生成建表语句”的：把 ORM 模型编译成某一种数据库方言的 DDL，
# 从而断言表结构本身长什么样。
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import mysql

import agent_demo.agent as agent_module
from agent_demo.agent import AgentApp
from agent_demo.memory import UserMemory
from agent_demo.structured_memory import (
    FactChange, MAX_FACTS, MemoryPatch, extract_changes, fact_status, merge_profile,
)

# 固定一个“当前时间”给测试用。带 +08:00 时区，是因为时区信息会参与断言。
NOW = datetime(2026, 10, 13, 14, 0, tzinfo=timezone(timedelta(hours=8)))


# ---------------------------------------------------------------------------
# 小工具：造一条事实变化
# ---------------------------------------------------------------------------
def change(key="姓名", value="晨星", action="upsert", expires_on=None):
    return FactChange(key=key, value=value, action=action, evidence="虚构测试原话", expires_on=expires_on)


# ---------------------------------------------------------------------------
# 合并：新值覆盖、未提到的保留、来源可追溯，而且不能改坏原画像
# ---------------------------------------------------------------------------
def test_updates_merge_without_losing_other_fields_and_keep_provenance():
    old = merge_profile({}, [change(), change("学习目标", "学习 Python")], 1, NOW)
    updated = merge_profile(old, [change(value="小舟")], 2, NOW)

    assert updated["姓名"]["value"] == "小舟"
    # 来源指向“这一次”的问答编号。
    assert updated["姓名"]["source_turn_id"] == 2
    # 记录时间带了时区偏移（+08:00），说明存的是带时区的时间。
    assert updated["姓名"]["recorded_at"].endswith("+08:00")
    # 这一轮没提到学习目标，它必须原样保留。
    assert updated["学习目标"] == old["学习目标"]
    assert old["姓名"]["value"] == "晨星"  # 不原地修改调用方的画像。


# ---------------------------------------------------------------------------
# 同一个事实再确认一次：复用字段名、只刷新时间，不新增字段
# ---------------------------------------------------------------------------
def test_same_fact_reuses_key_and_refreshes_confirmation_time():
    old = merge_profile({}, [change()], 1, NOW)

    # replace 是 datetime 自带的方法，用来“照原样复制、只改其中一部分”。
    later = NOW.replace(day=14)
    updated = merge_profile(old, [change()], 2, later)

    # 字段数没有增加 —— 说明“姓名”和“姓名”被认成了同一个字段。
    assert len(updated) == 1
    assert updated["姓名"]["source_turn_id"] == 2
    assert updated["姓名"]["recorded_at"].startswith("2026-10-14")
    # 什么都没变时，合并结果应当和原来等价（返回的是一份等值的新字典）。
    assert merge_profile(old, [], 3, later) == old


# ---------------------------------------------------------------------------
# 遗忘：留下标记、清空值；用户重新提供事实时可以恢复
# ---------------------------------------------------------------------------
def test_forget_keeps_marker_without_value_and_explicit_new_fact_can_restore(memory_chat):
    old = merge_profile({}, [change()], 1, NOW)
    forgotten = merge_profile(old, [change(value="", action="forget")], 2, NOW)

    assert forgotten["姓名"]["value"] is None
    assert fact_status(forgotten["姓名"], NOW.date()) == "已遗忘"
    # 直接检查 chat() 真正交给模型的画像：保留标记，不带 value。
    app, factory, _ = memory_chat
    app.memory.profile.return_value = forgotten
    app.chat("study-01", "我叫什么名字？")
    facts = json.loads(factory.return_value.invoke.call_args.args[0]["messages"][0].text.splitlines()[-1])  # 无摘要时画像是第一条资料消息。
    assert "value" not in facts["姓名"]
    assert facts["姓名"]["status"] == "已遗忘"
    assert forgotten["姓名"]["value"] is None

    # 用户重新明确说一次，就能恢复成有效事实。
    restored = merge_profile(forgotten, [change(value="小舟")], 3, NOW)
    assert fact_status(restored["姓名"], NOW.date()) == "有效"

    # 想遗忘一个本来就不存在的字段，等于什么都没发生，画像保持空。
    assert merge_profile({}, [change(value="", action="forget")], 4, NOW) == {}


# ---------------------------------------------------------------------------
# 有效期：含当天有效；过期后值不再进入上下文，但数据库记录不删
# ---------------------------------------------------------------------------
def test_expiry_is_inclusive_and_expired_values_are_not_injected(memory_chat, monkeypatch):
    # 有效期就设成“今天”。
    profile = merge_profile({}, [change("临时计划", "学习测试", expires_on=NOW.date())], 1, NOW)

    # 当天仍然有效 —— 这就是“含当天”的意思（判断用的是 > 而不是 >=）。
    assert fact_status(profile["临时计划"], NOW.date()) == "有效"
    # 第二天才算过期。
    assert fact_status(profile["临时计划"], date(2026, 10, 14)) == "已过期"

    app, factory, _ = memory_chat
    app.memory.profile.return_value = profile
    clock = Mock()
    monkeypatch.setattr(agent_module, "datetime", clock)

    clock.now.return_value = NOW
    app.chat("study-01", "我的计划是什么？")
    facts = json.loads(factory.return_value.invoke.call_args.args[0]["messages"][0].text.splitlines()[-1])  # 当天有效事实在画像消息中。
    assert facts["临时计划"] == {**profile["临时计划"], "status": "有效"}

    clock.now.return_value = NOW.replace(day=14)
    app.chat("study-01", "我的计划是什么？")
    facts = json.loads(factory.return_value.invoke.call_args.args[0]["messages"][0].text.splitlines()[-1])  # 次日画像隐藏失效值。
    # 第二天只隐藏 value，状态、来源和日期仍交给模型。
    assert facts["临时计划"] == {
        "recorded_at": NOW.isoformat(timespec="seconds"),
        "expires_on": "2026-10-13", "source_turn_id": 1,
        "forgotten": False, "status": "已过期",
    }
    assert profile["临时计划"]["value"] == "学习测试"  # 过期无需后台删除任务。


# ---------------------------------------------------------------------------
# 稳定事实不该被硬塞一个过期时间；年龄也不能被“推算”
# ---------------------------------------------------------------------------
def test_stable_fact_has_no_arbitrary_expiry_and_age_is_not_recalculated(memory_chat, monkeypatch):
    profile = merge_profile({}, [change("自述年龄", "28.5 岁")], 1, NOW)

    # 四年后依然有效 —— 因为姓名、年龄这类事实没有被凭空设过期日。
    assert fact_status(profile["自述年龄"], date(2030, 1, 1)) == "有效"

    app, factory, _ = memory_chat
    app.memory.profile.return_value = profile
    clock = Mock()
    clock.now.return_value = NOW.replace(year=2030)
    monkeypatch.setattr(agent_module, "datetime", clock)
    app.chat("study-01", "我之前说过几岁？")
    context = factory.return_value.invoke.call_args.args[0]["messages"][0].text  # 年龄与记录日期随画像资料发送。
    # 保留事实当时的记录日期；当前时间需要通过时间工具查询。
    assert "28.5 岁" in context and "2026-10-13" in context
    assert "2030-10-13" not in context
    assert profile["自述年龄"]["expires_on"] is None


def test_profile_text_is_stable_until_facts_or_status_change(memory_chat, monkeypatch):
    app, factory, _ = memory_chat
    profile = app.memory.profile.return_value
    clock = Mock()
    monkeypatch.setattr(agent_module, "datetime", clock)
    for current in (NOW, NOW + timedelta(seconds=45), NOW + timedelta(days=1)):
        clock.now.return_value = current
        app.chat("study-01", "我叫什么名字？")

    texts = [call.args[0]["messages"][0].text for call in factory.return_value.invoke.call_args_list]  # 三轮画像内容不因秒级时间改变。
    assert texts[0] == texts[1] == texts[2]
    text = texts[0]
    facts = json.loads(text.splitlines()[-1])
    assert facts == {"姓名": {**profile["姓名"], "status": "有效"}}
    assert "status" not in profile["姓名"]


def test_empty_profile_text_contains_only_an_empty_profile(memory_chat):
    app, factory, _ = memory_chat
    app.memory.profile.return_value = {}
    app.chat("study-01", "你好")
    text = factory.return_value.invoke.call_args.args[0]["messages"][0].text  # 空画像仍是独立资料消息。
    assert isinstance(text, str)
    assert json.loads(text.splitlines()[-1]) == {}


# ---------------------------------------------------------------------------
# 容量上限：宁可报错，也不偷偷丢掉旧字段
# ---------------------------------------------------------------------------
def test_profile_capacity_rejects_growth_without_silently_evicting_facts():
    old = merge_profile({}, [change(f"测试字段{i}") for i in range(MAX_FACTS)], 1, NOW)

    with pytest.raises(ValueError, match="最多保存"):
        merge_profile(old, [change("超限字段")], 2, NOW)

    # 原画像一个字段都没少 —— 采用“报错”而不是“淘汰最旧的”这种隐式行为。
    assert len(old) == MAX_FACTS


# ---------------------------------------------------------------------------
# 提取结果的业务校验：这些“格式合法但内容不合理”的补丁都要被拒
# ---------------------------------------------------------------------------
# 五种情况，分别是：
#   evidence 不是用户原话    → 疑似编造
#   同一个 key 出现两次      → 自相矛盾
#   upsert 但值为空          → 无意义的新增
#   forget 却带了值          → 语义冲突
#   forget 却带了有效期      → 语义冲突
# model_copy(update={...}) 是 pydantic 的“复制一份并改掉某个字段”，
# 用来在已有对象的基础上快速造一个变体。
@pytest.mark.parametrize("changes", [
    [change().model_copy(update={"evidence": "不是本轮说的话"})],
    [change(), change()],
    [change(value="")],
    [change(action="forget")],
    [change(action="forget", value="", expires_on=NOW.date())],
])
def test_extraction_rejects_unsupported_or_invalid_patches(changes):
    llm = Mock()
    # 伪造链条：with_structured_output(...).invoke(...) 直接返回我们想要的补丁。
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""), "parsed": MemoryPatch(changes=changes), "parsing_error": None,
    }

    changes, message, warning = extract_changes(llm, "虚构测试原话", {}, NOW)
    assert changes == [] and isinstance(message, AIMessage)
    assert "ValueError" in warning


# ---------------------------------------------------------------------------
# pydantic 层面的校验：多写字段、超长值都要被拦住
# ---------------------------------------------------------------------------
def test_schema_rejects_unknown_fields_and_oversized_values():
    # 多给了一个 invented 字段 → extra="forbid" 生效，报错。
    with pytest.raises(ValidationError):
        FactChange(key="姓名", action="upsert", value="晨星", evidence="原话", expires_on=None, invented=True)
    # value 超过 200 字上限 → 报错。
    with pytest.raises(ValidationError):
        change(value="a" * 201)


# ---------------------------------------------------------------------------
# 跑一遍真实的 LangChain 结构化输出链，只把最后的 HTTP 调用换掉
# ---------------------------------------------------------------------------
def test_real_langchain_structured_output_parses_mocked_function_call(monkeypatch):
    # 执行真实 LangChain/OpenAI 解析链，只替换发送 HTTP 的客户端方法。
    # 这个测试的价值在于：它验证的是“库真的会把函数调用解析成对象”，
    # 而不是我们自己写的解析逻辑 —— 所以不能用 Mock 把整条链都跳过。
    llm = ChatOpenAI(model="test-model", api_key="local-test-placeholder", use_responses_api=False)

    # 手工拼一份“服务端返回的原始响应”，格式和真实接口一致：
    # 里面带一个名为 MemoryPatch 的函数调用，参数就是序列化后的补丁 JSON。
    response = {
        "id": "test-response", "model": "test-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{
                "id": "test-call", "type": "function", "function": {
                    "name": "MemoryPatch", "arguments": MemoryPatch(changes=[change()]).model_dump_json(),
                },
            }],
        }}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    raw_response = Mock(headers={})
    raw_response.parse.return_value = response

    # client 本身就是个 Mock：调用它（发请求）就得到上面那份假响应。
    client = Mock(return_value=raw_response)

    # 只替换“真正发 HTTP 请求”的那一个方法，其余解析流程全是真实的。
    monkeypatch.setattr(llm.client.with_raw_response, "create", client)

    # 解析出来的结果应当和我们塞进去的补丁一致。
    changes, message, warning = extract_changes(llm, "虚构测试原话", {}, NOW)
    assert changes == [change()] and not warning
    assert message.usage_metadata["total_tokens"] == 2

    # 顺便验证发出去的请求内容也符合预期：
    request = client.call_args.kwargs
    # 用“函数调用”方式约束模型必须调 MemoryPatch 这个函数。
    assert request["tool_choice"]["function"]["name"] == "MemoryPatch"
    # 关掉思考过程参数确实带上了。
    assert request["extra_body"] == {"enable_thinking": False}
    # 用户原话被放进了最后一条消息里。
    assert "user_text" in request["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 画像表必须用 MySQL 原生 JSON 列，且 user_id 是主键
# ---------------------------------------------------------------------------
def test_profile_table_uses_native_mysql_json_and_a_user_primary_key():
    # 把 ORM 模型编译成 MySQL 的建表语句，直接检查表结构本身。
    ddl = str(CreateTable(UserMemory.__table__).compile(dialect=mysql.dialect()))

    # 用的是原生 JSON 类型（不是文本），而且不许为空。
    assert "facts_json JSON NOT NULL" in ddl
    # user_id 是主键 —— 这就是“一个用户只能有一行画像”的保证。
    assert "PRIMARY KEY (user_id)" in ddl
    # 画像表里不该有 session_id：画像是按用户共享的，和会话无关。
    assert "session_id" not in ddl


# ---------------------------------------------------------------------------
# 夹具：一个“半成品 app”，用假画像和假图跑完整 chat()
# ---------------------------------------------------------------------------
@pytest.fixture
def memory_chat(monkeypatch, settings):
    # 又是“造对象但跳过 __init__”，避免真去连库。
    app = AgentApp.__new__(AgentApp)
    app.settings = settings
    app.llm = Mock()
    app.memory = Mock()
    app.memory.context.return_value = ([], {}, {})  # 模拟新会话没有原文或摘要，用户共享画像仍可用。
    # 画像里预先放好“姓名=晨星”，来源标记为第 55 轮。
    app.memory.profile.return_value = merge_profile({}, [change()], 55, NOW)

    app.knowledge = Mock()
    app.knowledge.context.return_value = "知识库未找到足够相关的证据。"

    graph = Mock()
    graph.invoke.side_effect = lambda state, config: {"messages": state["messages"] + [AIMessage(content="晨星", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})]}  # 提供完整用量，避免把缺失值当作摘要已判断。
    factory = Mock(return_value=graph)

    # 替换掉真实的 Agent 装配与记忆提取，其余流程（含真实 chat 逻辑）照跑。
    monkeypatch.setattr(agent_module, "create_agent", factory)
    extractor = Mock(return_value=([], None, ""))
    monkeypatch.setattr(agent_module, "extract_changes", extractor)

    return app, factory, extractor


# ---------------------------------------------------------------------------
# 证据：即使当前会话没有原文或摘要，画像仍作为独立资料进入上下文。
# ---------------------------------------------------------------------------
def test_profile_is_loaded_without_recent_messages(memory_chat):
    app, factory, extractor = memory_chat
    result = app.chat("study-01", "我叫什么名字？")

    # 名字和来源编号都出现在独立画像消息里，固定系统规则不混入个人资料。
    assert "晨星" in factory.return_value.invoke.call_args.args[0]["messages"][0].text  # 姓名位于独立画像消息。
    assert '"source_turn_id": 55' in factory.return_value.invoke.call_args.args[0]["messages"][0].text  # 保留来源编号。
    # 近期聊天是空的（模拟原文已出窗口）。
    assert app.memory.context.return_value == ([], {}, {})  # 新会话不复制其他会话的原文或摘要。
    # 轨迹里只有那次自动知识检索，画像读取不算模型调用工具。
    assert [item["tool"] for item in result["trace"]] == ["knowledge_search"]
    # 提取记忆这一步照旧执行一次。
    extractor.assert_called_once()


# ---------------------------------------------------------------------------
# 换会话也能读到同一份画像，而且不复制聊天记录
# ---------------------------------------------------------------------------
def test_new_session_loads_the_same_user_profile_without_copying_chat_history(memory_chat):
    app, factory, _ = memory_chat
    app.chat("session-a", "我叫什么名字？")
    app.chat("session-b", "我叫什么名字？")

    # 两轮各读了一次画像。
    assert app.memory.profile.call_count == 2
    # 关键：读画像时一个参数都没传（没有 session_id）—— 证明画像只看用户。
    assert all(not call.args for call in app.memory.profile.call_args_list)
    # 两次拼出的系统提示词里都带着同一条姓名。
    assert all("晨星" in call.args[0]["messages"][0].text for call in factory.return_value.invoke.call_args_list)  # 两个会话都读取共享画像。
    # 而聊天上下文是按各自会话分别取的，存也是存到各自的会话里。
    app.memory.context.assert_called_with("session-b")
    assert app.memory.save_turn.call_args.args[0] == "session-b"


# ---------------------------------------------------------------------------
# 提取到变化时，问答和画像补丁要一起交给保存
# ---------------------------------------------------------------------------
def test_successful_answer_saves_profile_patch_and_original_turn_together(memory_chat):
    app, _, extractor = memory_chat
    extractor.return_value = ([change(value="小舟")], None, "")
    result = app.chat("study-01", "更正：我叫小舟")

    # save_turn 的第 5 个参数（下标 4）正是提取出来的变化列表。
    assert app.memory.save_turn.call_args.args[4] == extractor.return_value[0]
    # 第 2 个参数（下标 1）存的是用户原话。
    assert app.memory.save_turn.call_args.args[1] == "更正：我叫小舟"
    # 一切正常时不该有警告。
    assert not result["memory_warning"]


# ---------------------------------------------------------------------------
# 提取失败：原始问答照存，但要有可见警告，且不能泄露异常内容
# ---------------------------------------------------------------------------
def test_extraction_failure_preserves_raw_chat_and_has_visible_warning(memory_chat):
    app, _, extractor = memory_chat
    extractor.return_value = ([], None, "结构化记忆提取失败（RuntimeError），本轮未更新画像。")
    result = app.chat("study-01", "虚构测试原话")

    # 画像变化是空的，但问答依然保存了。
    assert app.memory.save_turn.call_args.args[4] == []
    # 警告里带上异常类型名，方便排查。
    assert "RuntimeError" in result["memory_warning"]
    # 但不带异常的具体内容 —— 因为它可能包含请求正文甚至密钥。
    assert "内容不能打印" not in result["memory_warning"]


# ---------------------------------------------------------------------------
# 画像超容量：事务回滚后，退一步只保存问答
# ---------------------------------------------------------------------------
def test_capacity_failure_saves_raw_turn_after_transaction_rollback(memory_chat):
    app, _, extractor = memory_chat
    extractor.return_value = ([change()], None, "")
    # side_effect 给列表：第一次调用抛错，第二次正常返回。
    # 正好模拟“第一次带画像保存失败 → 第二次只存问答成功”。
    app.memory.save_turn.side_effect = [ValueError("测试容量超限"), None]

    result = app.chat("study-01", "虚构测试原话")

    # 保存被尝试了两次。
    assert app.memory.save_turn.call_count == 2
    # 第二次只传了 4 个参数（没有画像变化）。
    assert len(app.memory.save_turn.call_args.args) == 4
    assert app.memory.save_turn.call_args.kwargs["usage"] == result["usage"]
    assert result["memory_warning"]


# ---------------------------------------------------------------------------
# 模型没给出有效回答时，压根不该开始提取记忆，更不该保存
# ---------------------------------------------------------------------------
def test_failed_answer_never_starts_memory_extraction(memory_chat):
    app, factory, extractor = memory_chat
    factory.return_value.invoke.side_effect = RuntimeError("模拟对话请求失败")

    with pytest.raises(RuntimeError):
        app.chat("study-01", "虚构测试原话")

    # 没有有效回答，就既不该提取记忆，也不该留记录。
    extractor.assert_not_called()
    app.memory.save_turn.assert_not_called()


def test_extraction_service_failure_returns_warning_without_secret():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("测试秘密")
    changes, message, warning = extract_changes(llm, "测试", {}, NOW)
    assert changes == [] and message is None
    assert "RuntimeError" in warning and "测试秘密" not in warning
