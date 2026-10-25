"""验证真实页面的会话交互；仅使用虚构内存数据，不连接模型或数据库。"""

# ---------------------------------------------------------------------------
# 标准库 / 第三方库导入
# ---------------------------------------------------------------------------
import json
from datetime import datetime

# SimpleNamespace 是个“能随手挂属性的空对象”，用来在测试里临时冒充一个类实例。
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# streamlit 本身也在测试里用到 —— 只是为了调它的缓存清理接口。
import streamlit as st

# AppTest 是 Streamlit 官方提供的测试工具：
# 它能真的把 app.py 跑起来，然后像用户那样点按钮、填输入框、读页面上的文字。
# 所以这个文件测的是“页面交互”，而不是某个函数。
from streamlit.testing.v1 import AppTest

# 注意这两个都 import 了“模块本身”：
# 因为下面要用 monkeypatch 替换模块里的名字（AgentApp 和 load_settings）。
import agent_demo.agent as agent_module
import agent_demo.config as config_module
import agent_demo.logging_config as log_module
from agent_demo.config import ROOT
from agent_demo.memory import validate_session_id


# ---------------------------------------------------------------------------
# 夹具：把真实页面跑起来，但把模型和数据库换成假的
# ---------------------------------------------------------------------------
@pytest.fixture
def session_page(monkeypatch, settings):
    # 页面行为测试不写项目真实日志；日志落盘由 test_logging.py 单独验证。
    monkeypatch.setattr(log_module, "setup_logging", Mock())
    # records 就是这份测试用的“假数据库”：一个字典，会话 ID → 聊天记录列表。
    records = {"saved-study": [{
        "user_text": "虚构历史问题", "assistant_text": "虚构历史回答", "trace_json": "[]", "usage_json": None,
    }]}

    memory = Mock()
    memory.profile.return_value = {}

    # side_effect 传函数时，每次调用都会执行它、用返回值当结果。
    # 这两行模拟了真实的读取行为：
    #   recent(sid, limit)   → 取该会话记录，再切出最后 limit 条
    #   list_sessions()      → 把有记录的会话汇总成“会话 ID + 轮数 + 时间”
    def recent(sid, limit=50):
        # 与真实数据库入口一致：显示文字不能被当成会话 ID 使用。
        validate_session_id(sid)
        return records.get(sid, [])[-limit:]
    memory.recent.side_effect = recent
    memory.list_sessions.side_effect = lambda: [{
        "session_id": sid, "turn_count": len(rows), "last_saved_at": datetime(2026, 1, 1),
    } for sid, rows in records.items() if rows]

    def chat(sid, question):
        # 假的对话：把这一轮追加进“假数据库”，并返回页面需要的结果字典。
        records.setdefault(sid, []).append({
            "user_text": question, "assistant_text": "虚构新回答", "trace_json": "[]", "usage_json": None,
        })
        return {"answer": "虚构新回答", "trace": []}

    # SimpleNamespace 冒充 AgentApp：页面只用到 settings、memory、chat、knowledge 这几个成员。
    knowledge = Mock()
    knowledge.ingest.return_value = 12
    knowledge.upload.return_value = {"files": ["lesson.txt"], "chunks": 13}
    app = SimpleNamespace(
        settings=settings, memory=memory, knowledge=knowledge, chat=Mock(side_effect=chat),
    )

    # 在页面导入前替换构造入口，避免 AgentApp 初始化读取真实知识库或调用 API。
    # 关键：app.py 里写的是 AgentApp(load_settings())，
    # 所以要把“构造 AgentApp”和“读配置”这两个入口都换成假的，
    # 页面跑起来时才会用上面那个简单的假对象。
    monkeypatch.setattr(agent_module, "AgentApp", lambda settings: app)
    monkeypatch.setattr(config_module, "load_settings", lambda: settings)

    # 清掉 Streamlit 的资源缓存，确保页面这次真的会重新构造（而不是复用上一次的缓存）。
    st.cache_resource.clear()
    try:
        # 真的跑一遍 app.py，default_timeout=5 表示每个动作最多等 5 秒。
        page = AppTest.from_file(ROOT / "app.py", default_timeout=5).run()
        # 页面初始化阶段就不该有异常。
        assert not page.exception
        # yield 把两样东西交给测试函数：页面对象 和 那个假 app。
        yield page, app
    finally:
        # 测试结束后再清一次，免得污染后面的测试。
        st.cache_resource.clear()


def test_summary_caption_uses_saved_cursor_without_changing_chat_history(session_page):  # 页面显示摘要状态，不按旧窗口限制模型历史。
    page, app = session_page  # 使用假数据库和真实 Streamlit 页面。
    app.memory.recent("saved-study")[-1]["summary_json"] = {"text": "更早历史摘要", "through_turn_id": 37}  # 给已保存会话添加摘要快照。
    page.selectbox(key="sid").select("saved-study").run()  # 切回该会话，让页面重新读取最后一行。
    assert not page.exception  # 新字段不能破坏聊天绘制。
    assert any("摘要已覆盖至对话 #37" in item.value for item in page.caption)  # 状态来自数据库真实覆盖 ID。
    assert any("上下文使用累计摘要与其后全部问答" in item.value for item in page.caption)  # 展示文字与新实现一致。
    assert any("虚构历史回答" in item.value for item in page.markdown)  # 原始聊天仍能在页面查看。


# ---------------------------------------------------------------------------
# 侧边栏：折叠布局、上传与同步
# ---------------------------------------------------------------------------
def test_usage_is_shown_per_turn_and_totals_survive_reruns_and_session_switches(session_page):
    page, app = session_page
    app.chat("saved-study", "第二轮测试")
    rows = app.memory.recent("saved-study")
    for row, stats in zip(rows, [
        {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
         "cached_tokens": 100, "reasoning_tokens": 10, "chat_seconds": 2.5},
        {"input_tokens": 900, "output_tokens": 90, "total_tokens": 990,
         "cached_tokens": 0, "reasoning_tokens": 0, "chat_seconds": 3.5},
    ]):
        before = rows[0]["usage_json"]["total"] if row is rows[1] else {}
        total = {key: value + before.get(key, 0) for key, value in stats.items()}
        row["usage_json"] = {"current": stats, "total": {**total, "turns": 2 if before else 1}}

    page.selectbox(key="sid").select("saved-study").run()
    assert not page.exception
    captions = "\n".join(item.value for item in page.caption)
    assert "本轮输入token：100 · 本轮输出token：10 · 本轮总token：110" in captions
    assert "本轮缓存命中率：100.00% · 本轮推理token占比：100.00% · 本轮处理耗时：2.50 秒" in captions
    assert "累计输入token：1,000 · 累计输出token：100 · 累计总token：1,100" in captions
    assert "累计缓存命中率：10.00% · 累计推理token占比：10.00% · 累计处理耗时：6.00 秒" in captions
    assert not any(item.label == "工具调用轨迹与知识来源" for item in page.expander)

    page.run()
    assert any("累计总token：1,100" in item.value for item in page.caption)
    page.button(key="new_session_clicked").click().run()
    assert any("累计用量：未记录" in item.value for item in page.caption)
    assert any(item.value == "**当前会话累计用量**" for item in page.markdown)
    page.selectbox(key="sid").select("saved-study").run()
    assert any("累计总token：1,100" in item.value for item in page.caption)


def test_usage_distinguishes_missing_fields_from_explicit_zero(session_page):
    page, app = session_page
    row = app.memory.recent("saved-study")[0]
    stats = {
        "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
        "cached_tokens": 0, "reasoning_tokens": None, "chat_seconds": 1.0,
    }
    row["usage_json"] = {"current": stats, "total": {**stats, "turns": 1}}
    page.selectbox(key="sid").select("saved-study").run()
    assert not page.exception
    assert any("本轮缓存命中率：0.00% · 本轮推理token占比：未提供" in item.value for item in page.caption)


def test_sidebar_groups_optional_controls_without_starting_operations(session_page):
    """知识库管理默认收起，运行信息直接显示；画按钮不会触发后台操作。"""
    page, app = session_page
    assert [item.label for item in page.sidebar.expander] == ["知识库管理"]
    assert all(not item.proto.expanded for item in page.sidebar.expander)
    captions = "\n".join(item.value for item in page.sidebar.caption)
    assert f"当前用户：{app.settings.user_id}" in captions
    assert f"问答模型：{app.settings.llm_model}" in captions
    assert f"向量模型：{app.settings.embedding_model}" in captions
    assert f"数据库：{app.settings.mysql_database}" in captions
    assert page.selectbox(key="sid")
    assert page.button(key="new_session_clicked")
    app.knowledge.upload.assert_not_called()
    app.knowledge.ingest.assert_not_called()
    app.chat.assert_not_called()


def test_sidebar_upload_passes_file_bytes_to_handler_once(session_page):
    page, app = session_page
    page.file_uploader[0].set_value([("lesson.txt", b"test lesson", "text/plain")]).run()
    app.knowledge.upload.assert_not_called()

    next(button for button in page.button if button.label == "上传并入库").click().run()
    assert not page.exception
    app.knowledge.upload.assert_called_once_with([("lesson.txt", b"test lesson")])
    assert any("已入库 1 份文档" in item.value for item in page.sidebar.success)
    assert page.chat_input
    page.run()
    assert app.knowledge.upload.call_count == 1


def test_sidebar_sync_runs_once_and_keeps_chat_available(session_page):
    page, app = session_page
    next(button for button in page.button if button.label == "同步知识库").click().run()
    assert not page.exception
    app.knowledge.ingest.assert_called_once_with()
    assert any("12 个切片" in item.value for item in page.sidebar.success)
    assert page.chat_input
    page.run()
    assert app.knowledge.ingest.call_count == 1


@pytest.mark.parametrize("action", ["upload", "sync"])
def test_sidebar_operation_failure_keeps_chat_available(session_page, action):
    page, app = session_page
    if action == "upload":
        page.file_uploader[0].set_value([("lesson.txt", b"test lesson", "text/plain")]).run()
        app.knowledge.upload.side_effect = ValueError("虚构上传失败")
        label = "上传并入库"
    else:
        app.knowledge.ingest.side_effect = ValueError("虚构同步失败")
        label = "同步知识库"
    next(button for button in page.button if button.label == label).click().run()
    assert not page.exception
    assert page.sidebar.error
    assert page.chat_input
    app.chat.assert_not_called()


# ---------------------------------------------------------------------------
# 选中历史会话 → 直接切换 → 能看到历史记录
# ---------------------------------------------------------------------------
def test_saved_session_selection_switches_immediately(session_page):
    """选中历史就能看到聊天，不再经过手工 ID 和加载按钮。"""
    page, app = session_page
    page.selectbox(key="sid").select("saved-study").run()

    assert not page.exception
    assert page.session_state["sid"] == "saved-study"
    assert any("已切换会话" in item.value for item in page.success)
    assert any(item.value == "虚构历史回答" for item in page.markdown)
    assert not page.text_input
    app.chat.assert_not_called()


# ---------------------------------------------------------------------------
# 还没有历史会话时，当前编号仍然能显示和使用
# ---------------------------------------------------------------------------
def test_empty_history_keeps_new_session_selectable(session_page):
    """还没有保存过聊天时，下拉框仍能显示当前会话并继续提问。"""
    page, app = session_page
    assert len(page.sidebar.info) == 1
    assert "可以直接提问" in page.sidebar.info[0].value
    sid = page.session_state["sid"]
    app.memory.list_sessions.side_effect = lambda: []
    page.run()

    assert not page.exception
    assert page.selectbox(key="sid").value == sid
    # 普通重跑后，默认说明仍显示在侧边栏。
    assert len(page.sidebar.info) == 1
    assert "可以直接提问" in page.sidebar.info[0].value
    page.chat_input[0].set_value("第一轮问题").run()
    app.chat.assert_called_once_with(sid, "第一轮问题")


# ---------------------------------------------------------------------------
# 页面重跑时，当前选择保持稳定
# ---------------------------------------------------------------------------
def test_page_rerun_keeps_selected_session(session_page):
    """普通重跑不会生成新编号，也不会切回其他会话。"""
    page, app = session_page
    page.selectbox(key="sid").select("saved-study").run()
    page.run()

    assert not page.exception
    assert page.session_state["sid"] == page.selectbox(key="sid").value == "saved-study"
    assert any(item.value == "虚构历史回答" for item in page.markdown)
    assert len(page.sidebar.success) == 1
    assert "已切换会话" in page.sidebar.success[0].value
    app.chat.assert_not_called()


# ---------------------------------------------------------------------------
# 新建会话：换 ID、下拉框同步显示，但历史记录一条都不能丢
# ---------------------------------------------------------------------------
def test_new_session_preserves_saved_history_and_resets_selection(session_page):
    page, _ = session_page

    # 先加载一个旧会话，制造出“当前有会话”的状态。
    page.selectbox(key="sid").select("saved-study").run()

    # 再点“新建会话”。
    page.button(key="new_session_clicked").click().run()
    assert not page.exception

    # 会话 ID 换成了新的。
    assert page.session_state["sid"] != "saved-study"
    # 下拉框与后台都显示同一个新 ID。
    assert page.selectbox(key="sid").value == page.session_state["sid"]
    # 有蓝色提示告诉用户“聊完才会自动保存”。
    assert any("完成一轮问答后自动保存" in item.value for item in page.info)

    # 最关键的一条：旧会话仍然出现在下拉框的候选项里 —— 记录没被删掉。
    # ["saved-study"[:12] 是取 ID 前 12 位，因为下拉框显示的是截短后的文字。
    assert any("saved-study"[:12] in label for label in page.selectbox(key="sid").options)


# ---------------------------------------------------------------------------
# 说完一轮之后，应该自动保存，并且能被重新加载出来
# ---------------------------------------------------------------------------
def test_completed_chat_is_saved_and_available_to_load(session_page):
    page, app = session_page
    sid = page.session_state["sid"]

    # 往底部输入框里打字并发送。chat_input[0] 表示“页面上第一个聊天输入框”。
    page.chat_input[0].set_value("虚构新问题").run()
    assert not page.exception

    # 页面确实把问题转发给了 Agent，用的也是当前会话 ID。
    app.chat.assert_called_once_with(sid, "虚构新问题")
    # 并且显示“已自动保存”的绿框提示。
    assert any("自动保存到 MySQL" in item.value for item in page.success)

    # 现在这个会话应该有记录了，可以选出来重新加载。
    page.selectbox(key="sid").select("saved-study").run()
    page.selectbox(key="sid").select(sid).run()
    assert not page.exception
    assert any(item.value == "虚构新回答" for item in page.markdown)


def test_second_message_keeps_session_when_browser_resubmits_previous_label(session_page):
    """模拟浏览器还带着第一轮之前的选中值，不能把旧显示文字写成会话 ID。"""
    page, app = session_page
    sid = page.session_state["sid"]
    selector = page.selectbox(key="sid")
    old_value = selector._widget_state.string_value
    page.chat_input[0].set_value("第一轮测试").run()
    assert not page.error

    page.chat_input[0].set_value("第二轮测试")
    states = page._tree.get_widget_states()
    selector_id = page.selectbox(key="sid").id
    next(widget for widget in states.widgets if widget.id == selector_id).string_value = old_value
    page._run(states)

    assert not page.exception and not page.error
    assert page.session_state["sid"] == sid
    assert app.chat.call_args.args == (sid, "第二轮测试")
    assert len(app.memory.recent(sid)) == 2


# ---------------------------------------------------------------------------
# 提问失败时，绝不能残留上一轮“已保存”的成功提示
# ---------------------------------------------------------------------------
def test_failed_chat_does_not_keep_the_previous_save_success_notice(session_page):
    page, app = session_page
    sid = page.session_state["sid"]

    # 先正常问一轮，页面上会出现“已保存”的绿框。
    page.chat_input[0].set_value("虚构成功问题").run()

    # 再让对话抛错，然后问第二轮。
    app.chat.side_effect = RuntimeError("模拟请求失败")
    page.chat_input[0].set_value("虚构失败问题").run()

    assert not page.exception
    # 应当有红框错误。
    assert page.error
    # 而且不能再有“已保存”的绿框 —— 这一轮根本没保存成功，不能谎报。
    assert not any("自动保存到 MySQL" in item.value for item in page.success)
    # 失败后回到默认说明，不留下上次的成功提示，也不需要清空占位控件。
    assert len(page.sidebar.info) == 1
    assert "可以直接提问" in page.sidebar.info[0].value
    # 假数据库里也确实只存下了成功的那一轮。
    assert len(app.memory.recent(sid)) == 1


# ---------------------------------------------------------------------------
# 共享画像在任何会话里都看得见
# ---------------------------------------------------------------------------
def test_shared_profile_is_visible_after_loading_and_creating_sessions(session_page):
    page, app = session_page

    # 给假画像塞一条“姓名=晨星”。
    app.memory.profile.return_value = {"姓名": {
        "value": "晨星", "recorded_at": "2026-01-01T12:00:00+08:00",
        "expires_on": None, "source_turn_id": 1, "forgotten": False,
    }}

    page.selectbox(key="sid").select("saved-study").run()
    assert not page.exception

    # 页面上的 JSON 展示里应当有这条事实，并且状态被现算成“有效”。
    assert any(json.loads(item.value).get("姓名", {}).get("value") == "晨星"
               and json.loads(item.value)["姓名"]["status"] == "有效" for item in page.json)

    # 读画像时不该传任何参数（画像只看用户，不看会话）。
    app.memory.profile.assert_called_with()

    # 换成新会话，画像依然在 —— 这就是“跨会话共享”。
    page.button(key="new_session_clicked").click().run()
    assert not page.exception
    assert page.session_state["sid"] != "saved-study"
    assert any(json.loads(item.value).get("姓名", {}).get("value") == "晨星" for item in page.json)
    # 侧边栏也显示了当前用户是谁。
    assert any("当前用户：local_user" in item.value for item in page.caption)


# ---------------------------------------------------------------------------
# 记忆提取失败：要有黄框警告，但问答本身仍然保存
# ---------------------------------------------------------------------------
def test_memory_extraction_failure_is_visible_but_chat_remains_saved(session_page):
    page, app = session_page
    original_chat = app.chat.side_effect

    def chat(sid, question):
        # 先走原本的假对话拿到结果，再用解包造一份新字典、补上 memory_warning。
        result = original_chat(sid, question)
        return {**result, "memory_warning": "结构化记忆提取失败，本轮未更新画像。"}

    app.chat.side_effect = chat

    sid = page.session_state["sid"]
    page.chat_input[0].set_value("虚构测试原话").run()
    assert not page.exception

    # 黄框提示里要同时体现“已保存”和“提取失败”两件事 ——
    # 因为它们本来就是互不影响的：问答存住了，只是画像没更新。
    assert any("本轮问答已自动保存" in item.value and "提取失败" in item.value for item in page.warning)
    assert app.memory.recent(sid)


@pytest.mark.parametrize("status", ["passed", "revised", "review_failed", "revision_failed"])
def test_saved_reflection_is_visible_beside_legacy_tool_trace(session_page, status):
    page, app = session_page
    # 直接往假数据库的旧会话里放检查记录，再加载页面，验证展示来自已保存数据。
    row = app.memory.recent("saved-study")[0]
    row["trace_json"] = json.dumps([
        {"tool": "calculator", "args": {"expression": "1+2"}, "result": "3"},
        {"tool": "self_reflection", "kind": "reflection", "status": status,
         "draft": "虚构检查前草稿", "issues": [], "result": "虚构检查状态：" + status},
    ], ensure_ascii=False)

    page.selectbox(key="sid").select("saved-study").run()

    assert not page.exception
    assert any(item.value == "虚构检查前草稿" for item in page.markdown)
    assert any(json.loads(item.value) == [
        {"tool": "calculator", "args": {"expression": "1+2"}, "result": "3"},
    ] for item in page.json)
    messages = page.warning if status.endswith("failed") else page.caption
    assert any("虚构检查状态：" + status in item.value for item in messages)
    app.chat.assert_not_called()
