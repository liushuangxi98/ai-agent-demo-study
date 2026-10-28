"""Streamlit 页面：会话管理、知识库操作、聊天与工具轨迹展示。"""

import json
import logging
from datetime import date, datetime
from uuid import uuid4

import streamlit as st

from agent_demo.agent import AgentApp
from agent_demo.config import load_settings
from agent_demo.logging_config import LOG_FILE, setup_logging
from agent_demo.structured_memory import fact_status

logger = logging.getLogger("agent_demo.app")


@st.cache_resource
def get_app() -> AgentApp:
    """创建并缓存 Agent，避免每次页面重跑都重新连库和同步知识。"""
    return AgentApp(load_settings())


def show_error(error: Exception) -> None:
    """完整堆栈写日志；页面只显示简短提示。"""
    logger.exception("页面操作失败")
    detail = str(error) if isinstance(error, ValueError) else "请检查开发文档中的配置和网络说明。"
    st.error(f"{type(error).__name__}：{detail}")
    st.caption(f"详细日志：{LOG_FILE}")


def new_session() -> None:
    """换一个会话号；旧聊天和共享画像继续保留。"""
    st.session_state.sid = uuid4().hex
    st.session_state.session_notice = ("info", "已创建新会话。完成一轮问答后自动保存。")


def switch_session() -> None:
    """下拉框已自动更新 sid，这里只准备切换成功的提示。"""
    st.session_state.session_notice = ("success", "已切换会话，聊天记录显示在主区域。")


def handle_upload(app: AgentApp, files: list) -> None:
    """把选中的文件交给知识库，显示入库结果。"""
    uploads = [(file.name, file.getvalue()) for file in files]
    with st.spinner("读取文档并建立知识库…"):
        result = app.knowledge.upload(uploads)
    st.success(f"已入库 {len(result['files'])} 份文档，知识库共 {result['chunks']} 个切片")


def handle_sync(app: AgentApp) -> None:
    """重新读取知识原件、更新索引，显示切片总数。"""
    with st.spinner("切片与向量化中…"):
        count = app.knowledge.ingest()
    st.success(f"知识库已同步：{count} 个切片")


def handle_chat_result(result: dict) -> None:
    """根据本轮结果准备提示并重跑页面。"""
    if result.get("memory_warning"):
        st.session_state.session_notice = (
            "warning", "本轮问答已自动保存到 MySQL。" + (result.get("memory_warning") or "结构化记忆已检查更新。")
        )
    else:
        st.session_state.session_notice = ("success", "本轮问答已自动保存到 MySQL，结构化记忆已检查更新。")
    st.rerun()


def handle_question(app: AgentApp, question: str) -> None:
    """调用 Agent 回答问题，准备结果提示，再重画最新聊天。"""
    # 先清掉上次提示，避免本轮失败时仍显示上轮的“已保存”。
    st.session_state.session_notice = None
    with st.spinner("Agent 正在调用模型与工具…"):
        result = app.chat(st.session_state.sid, question)

    if result.get("status") == "pending_approval":
        st.session_state.pending_approval = result
        st.session_state.session_notice = ("warning", "等待人工审批：请先批准或拒绝待执行的操作。")
        st.rerun()

    handle_chat_result(result)


def resume_question(app: AgentApp, pending: dict, decision: dict) -> None:
    """批准或拒绝待审批操作，并继续本轮。"""
    try:
        with st.spinner("Agent 正在继续…"):
            result = app.resume(pending["session_id"], decision)
    except Exception as error:
        show_error(error)
        st.session_state.pending_approval = None
        return

    if result.get("status") == "pending_approval":
        st.session_state.pending_approval = result
        st.rerun()

    st.session_state.pending_approval = None
    handle_chat_result(result)


def render_approval(app: AgentApp, pending: dict) -> None:
    """显示待审批的工具调用，批准或拒绝后继续本轮。"""
    st.warning("以下工具调用需要人工审批，执行前不会修改任何数据。")
    st.json(pending["actions"])
    approve, reject = st.columns(2)
    approve_clicked = approve.button("批准执行", type="primary", width="stretch")
    reject_clicked = reject.button("拒绝", width="stretch")
    if approve_clicked:
        resume_question(app, pending, {"type": "approve"})
    elif reject_clicked:
        resume_question(app, pending, {"type": "reject", "message": "用户拒绝"})


def render_usage(stats: dict | None, prefix: str = "本轮") -> None:
    """两行显示六项指标；缺失显示“未提供”，明确返回 0 才是零消耗。"""
    if stats is None:
        st.caption(f"{prefix}用量：未记录。")
        return

    counts = {
        field: f"{stats[field]:,}" if stats.get(field) is not None else "未提供"
        for field in ("input_tokens", "output_tokens", "total_tokens")
    }
    # 比例现算：缓存 token / 输入 token，推理 token / 输出 token，不能平均每轮百分比。
    rates = {}
    for label, numerator, denominator in (
        ("缓存命中率", "cached_tokens", "input_tokens"),
        ("推理token占比", "reasoning_tokens", "output_tokens"),
    ):
        if stats.get(numerator) is None or stats.get(denominator) is None:
            rates[label] = "未提供"
        elif stats[denominator] == 0:
            rates[label] = "—"  # 分母为 0 时比例没有意义，也不能做除法。
        else:
            rates[label] = f"{stats[numerator] / stats[denominator]:.2%}"

    st.caption(
        f"{prefix}输入token：{counts['input_tokens']} · "
        f"{prefix}输出token：{counts['output_tokens']} · "
        f"{prefix}总token：{counts['total_tokens']}"
    )
    st.caption(
        f"{prefix}缓存命中率：{rates['缓存命中率']} · "
        f"{prefix}推理token占比：{rates['推理token占比']} · "
        f"{prefix}处理耗时：{stats['chat_seconds']:.2f} 秒"
    )
    if any(stats.get(field) is None for field in (
        "input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens",
    )):
        st.caption("部分请求未提供完整用量，缺失的项目不估算、不当作 0。")


def render_profile(profile: dict, today: date) -> None:
    """画共享画像面板；过期和遗忘状态按当天现算，不写回数据库。"""
    with st.expander("个人记忆（所有会话共享）"):
        if profile:
            st.json({
                key: {**fact, "status": fact_status(fact, today)}
                for key, fact in profile.items()
            })
        else:
            st.info("尚无结构化记忆。试试：我叫小明，我的学习目标是掌握 Python。")


def render_sidebar(app: AgentApp) -> tuple[list, bool, bool]:
    """画侧栏，返回文件列表、上传点击值和同步点击值，交给 main() 处理。"""
    st.header("会话")
    try:
        saved_sessions = {row["session_id"]: row for row in app.memory.list_sessions()}
    except Exception as error:
        show_error(error)
        st.stop()

    options = [*saved_sessions]
    # 新会话还没有入库，也要放进候选项，保证下拉框能显示当前编号。
    if st.session_state.sid not in saved_sessions:
        options.insert(0, st.session_state.sid)

    st.selectbox(
        "切换会话（最近 50 个）",
        options=options,
        # 选项文字保持完整 ID 且不随轮数变化，避免浏览器提交旧显示文字。
        key="sid",
        on_change=switch_session,
    )
    saved = saved_sessions.get(st.session_state.sid)
    if saved:
        st.caption(f"已保存 {saved['turn_count']} 轮 · 最近保存：{saved['last_saved_at']:%m-%d %H:%M}")
    st.button("新建会话", key="new_session_clicked", on_click=new_session, width="stretch")
    st.caption("问答自动保存；新建会话保留旧记录。")

    # 收起的抽屉只影响显示，里面的代码仍会执行。
    with st.expander("知识库管理", expanded=False):
        files = st.file_uploader(
            "上传知识文档", type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True, max_upload_size=10,
        )
        st.caption("单文件最多10MB；同名文件更新。")

        upload_column, sync_column = st.columns(2)
        upload_clicked = upload_column.button("上传并入库", disabled=not files, width="stretch")
        sync_clicked = sync_column.button("同步知识库", width="stretch")

    st.markdown("**运行信息**")
    st.caption(f"当前用户：{app.settings.user_id} · 新会话共享记忆")
    st.caption(f"问答模型：{app.settings.llm_model}")
    st.caption(f"向量模型：{app.settings.embedding_model}")
    st.caption(f"数据库：{app.settings.mysql_database}")

    return files, upload_clicked, sync_clicked


def render_chat(app: AgentApp) -> str | None:
    """显示画像和历史问答；提交时返回新问题，否则返回 None。"""
    st.caption(
        f"当前会话：{st.session_state.sid} · "
        "上下文使用累计摘要与其后全部问答 · 页面显示最近 50 轮"
    )
    st.caption("每次提问会先检索知识库。试试：计算 (12+8)*3 ｜ 现在几点 ｜ 星舟课程退款期限 ｜ 我叫小明 ｜ 回忆我的名字")

    today = datetime.now().astimezone().date()
    profile = app.memory.profile()
    render_profile(profile, today)

    rows = app.memory.recent(st.session_state.sid)
    summary_state = (rows[-1].get("summary_json") or {}) if rows else {}
    if summary_state:
        st.caption(f"历史摘要已覆盖至对话 #{summary_state['through_turn_id']}；之后的全部问答原文继续进入上下文。")
    for row in rows:
        with st.chat_message("user"):
            st.markdown(row["user_text"])

        with st.chat_message("assistant"):
            st.markdown(row["assistant_text"])
            render_usage((row.get("usage_json") or {}).get("current"))
            try:
                trace = json.loads(row["trace_json"])
            except (ValueError, TypeError):
                trace = []
            trace = [item for item in trace if isinstance(item, dict)] if isinstance(trace, list) else []
            tool_trace = [item for item in trace if item.get("kind") != "reflection"]
            if tool_trace:
                with st.expander("工具调用轨迹与知识来源"):
                    st.json(tool_trace)

            for note in (item for item in trace if item.get("kind") == "reflection"):
                with st.expander("答案检查"):
                    if note["status"] in {"review_failed", "revision_failed"}:
                        st.warning(note["result"])
                    else:
                        st.caption(note["result"])
                    if note["issues"]:
                        st.json(note["issues"])
                    st.markdown("**检查前的草稿**")
                    st.markdown(note["draft"])

            # 每轮结束后再显示一次，便于对照“这轮回答时”的最新画像。
            render_profile(profile, today)

    # 最后一轮已保存“截至这一轮”的累计，直接显示，不重算。
    latest_usage = (rows[-1].get("usage_json") or {}) if rows else {}
    total = latest_usage.get("total")
    st.markdown("**当前会话累计用量**")
    render_usage(total, prefix="累计")
    st.caption(
        f"已统计 {total['turns'] if total else 0} 轮；token 包含已保存问答的回答、反思和记忆提取，"
        "包括自动摘要，不含 Embedding 和未保存的失败问答。"
    )
    st.caption("缓存 token 仍计入输入；处理耗时从进入 chat() 到保存前，包含检索、工具、反思、记忆提取和自动摘要。")

    pending = st.session_state.get("pending_approval")
    if pending and pending.get("session_id") == st.session_state.sid:
        render_approval(app, pending)
        return None

    return st.chat_input("输入问题")


def main() -> None:
    """每次页面重跑：准备资源 → 侧栏 → 聊天 → 提示。"""
    # 先开日志，配置、数据库或知识库初始化失败也能留下堆栈。
    setup_logging()
    st.set_page_config(page_title="AI Agent 学习实验室", page_icon="🤖", layout="wide")
    st.html("""
        <style>
        [data-testid="stSidebarHeader"] {
            position: absolute;
            z-index: 1;
            top: 0;
            right: 1.25rem;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 0.5rem;
        }
        [data-testid="stSidebarUserContent"] h2 { padding-top: 0; }
        </style>
    """)
    st.title("🤖 AI Agent 学习实验室")

    try:
        app = get_app()
    except Exception as error:
        show_error(error)
        st.stop()

    if "sid" not in st.session_state:
        st.session_state.sid = uuid4().hex

    with st.sidebar:
        files, upload_clicked, sync_clicked = render_sidebar(app)
        try:
            if upload_clicked:
                handle_upload(app, files)
            if sync_clicked:
                handle_sync(app)
        except Exception as error:
            show_error(error)

    try:
        question = render_chat(app)
        if question:
            handle_question(app, question)
    except Exception as error:
        show_error(error)

    notice_type, notice_text = st.session_state.get("session_notice") or (
        "info", "可以直接提问，也可以切换会话或上传知识文档。"
    )
    show_notice = getattr(st.sidebar, notice_type)
    show_notice(notice_text)


if __name__ == "__main__":
    main()
