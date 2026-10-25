"""验证真实 ORM 的新增、修改、查询和事务回滚，不读取 .env 或接触用户数据库。

测试用内存 SQLite 执行 ORM，MySQL 特有的锁与并发另外验证；这不是应用的运行模式。
仅在测试中把 LONGTEXT 编译为 TEXT，并注册相同名称的大小写敏感排序规则。
"""

import json
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# 第三方库导入：SQLAlchemy
# ---------------------------------------------------------------------------
# create_engine / select / func：建引擎、构造查询、调用数据库函数
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

# 注意：真实运行用的是 MySQL 专属类型 LONGTEXT，而测试跑在 SQLite 上，
# 所以下面要“教”SQLAlchemy 在 SQLite 里该怎么处理它。
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import IntegrityError

# compiles 是装饰器：用来给某种类型注册“在某个数据库方言下应该编译成什么”。
from sqlalchemy.ext.compiler import compiles

from sqlalchemy.orm import sessionmaker

from agent_demo.memory import Base, ConversationMemory, ConversationTurn, UserMemory
from agent_demo.structured_memory import FactChange, MAX_FACTS
from agent_demo.usage import summarize_usage


# ---------------------------------------------------------------------------
# 让 SQLite 也能“看懂”MySQL 的 LONGTEXT
# ---------------------------------------------------------------------------
# @compiles(LONGTEXT, "sqlite") 的意思是：
# “当要把 LONGTEXT 这种类型编译到 sqlite 方言时，请用下面这个函数的结果”。
# 这只是为了测试能建表，真实的 MySQL 建表语句一点没变。
@compiles(LONGTEXT, "sqlite")
def compile_test_longtext(column_type, compiler, **kwargs):
    """仅让测试数据库接受 MySQL LONGTEXT 列；生产 MySQL 的表结构不变。"""
    return "TEXT"


# ---------------------------------------------------------------------------
# 夹具：搭一个“内存版数据库”，行为跟真库一样
# ---------------------------------------------------------------------------
@pytest.fixture
def memory(settings):
    """创建真实 ORM Session 和临时内存表，测试结束即释放，不调用生产初始化。"""
    # "sqlite://" 表示纯内存数据库：跑完就消失，不落一个文件，也不会碰到你的真库。
    engine = create_engine("sqlite://")

    # 项目里 user_id 用的是 utf8mb4_bin（大小写敏感）。
    # SQLite 默认没有这个排序规则，所以这里手工注册一个同名的、行为一致的规则，
    # 免得换个数据库跑，测试结果就变了。
    with engine.connect() as connection:
        connection.connection.driver_connection.create_collation(
            "utf8mb4_bin", lambda left, right: (left > right) - (left < right),
        )

    # 照着模型定义把表建出来。
    Base.metadata.create_all(engine)

    # 又是“造对象但跳过 __init__”：因为 __init__ 会去连真 MySQL，
    # 而这里我们只想用上面那个内存引擎。
    instance = ConversationMemory.__new__(ConversationMemory)
    instance.settings = settings
    instance.engine = engine
    instance.session_factory = sessionmaker(bind=engine)

    try:
        # 把搭好的对象交给测试使用。
        yield instance
    finally:
        # 无论测试成功失败，都要释放连接。
        instance.close()


# ---------------------------------------------------------------------------
# 小工具：造一条事实变化
# ---------------------------------------------------------------------------
def test_saved_snapshot_reuses_last_total_without_scanning_older_turns(memory):
    from langchain_core.messages import AIMessage
    message = AIMessage(content="回答", usage_metadata={
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
        "input_token_details": {"cache_read": 0}, "output_token_details": {"reasoning": 0},
    })
    # 超过旧的 6 轮窗口和页面显示的 50 轮，原文与累计都仍完整。
    for _ in range(55):
        history, previous, summary = memory.context("usage-session")  # 摘要状态与原文行、累计分开读取。
        stats = summarize_usage([message], previous, 0.5)
        memory.save_turn("usage-session", "问题", "回答", [], usage=stats)
    history, total, summary = memory.context("usage-session")  # 超过旧的 6/50 轮限制，仍读取全部未压缩问答。
    assert len(history) == 55 and summary == {}  # 没有摘要时不能默默丢弃早期问答。
    assert total["total_tokens"] == 660 and total["turns"] == 55
    assert total["chat_seconds"] == 27.5
    assert memory.recent("usage-session")[-1]["usage_json"]["total"] == total
    assert all(turn["user_text"] == "问题" and turn["assistant_text"] == "回答" for turn in history)  # 原文与统计保持不同字段。
    assert memory.context("other-session") == ([], {}, {})  # 另一会话的三个状态均为空。
    # 同名会话也不能混入其他用户的数据。
    memory.settings = replace(memory.settings, user_id="other-user")
    assert memory.context("usage-session") == ([], {}, {})  # 同名会话仍按用户隔离。


def test_context_preserves_long_original_turn_and_cumulative_usage(memory):  # 取消字符截断后，长问答也要完整交回。
    stats = {"current": {"total_tokens": 12}, "total": {"total_tokens": 1000, "turns": 9}}
    memory.save_turn("long-session", "很长的问题", "很长的回答", [], usage=stats)
    turns, total, summary = memory.context("long-session")  # 从数据库读取原文、累计与摘要。
    assert [(turn["user_text"], turn["assistant_text"]) for turn in turns] == [("很长的问题", "很长的回答")]  # 问答不截断。
    assert total == stats["total"] and summary == {}  # 没有摘要仍保留累计快照。
    assert memory.recent("long-session")[0]["usage_json"] == stats


def test_usage_json_is_a_nullable_native_mysql_json_column():
    ddl = str(CreateTable(ConversationTurn.__table__).compile(dialect=mysql.dialect()))
    assert "usage_json JSON" in ddl and "usage_json JSON NOT NULL" not in ddl
    assert "summary_json JSON" in ddl and "summary_json JSON NOT NULL" not in ddl  # 尚未摘要的问答使用 SQL NULL。


def test_summary_snapshot_reads_all_rows_after_real_cursor_and_survives_new_reader(memory):  # 覆盖位置不是会话轮数。
    memory.save_turn("session-a", "早期问题", "早期答案", [])  # 第一条待总结记录。
    memory.save_turn("session-b", "另一会话插入", "回答", [])  # 让全表 ID 在本会话中不连续。
    memory.save_turn("session-a", "第二个旧问题", "第二个旧答案", [])  # 第二条待总结记录。
    cursor = memory.recent("session-a")[-1]["id"]  # 从真实 ORM 记录取得摘要边界。
    state = {"text": "前两轮的摘要", "through_turn_id": cursor}  # 一份完整的累计摘要快照。
    for index in range(55):  # 摘要后也要读取超过页面 50 轮的全部原文。
        memory.save_turn("session-a", f"未压缩问题{index}", f"未压缩回答{index}", [], summary=state)  # 没重压缩时原样沿用快照。
    recovered = ConversationMemory.__new__(ConversationMemory)  # 新读写器不共享 Python 中的画像或聊天缓存。
    recovered.settings, recovered.engine = memory.settings, memory.engine  # 使用同一个测试内存数据库。
    recovered.session_factory = sessionmaker(bind=memory.engine)  # 创建新的 Session 工厂，模拟重新连接读取。
    turns, previous, summary = recovered.context("session-a")  # 重新从数据库恢复摘要和未压缩尾部。
    assert len(turns) == 55 and summary == state  # 快照恢复，未压缩原文没有被 6/50 轮限制。
    assert turns[0]["user_text"] == "未压缩问题0" and turns[-1]["user_text"] == "未压缩问题54"  # 原文顺序完整。
    assert len(memory.recent("session-a", limit=100)) == 57  # 被总结的旧原文仍保存在数据库。
    assert recovered.context("session-b")[2] == {}  # 另一会话不能读取本会话摘要。
    recovered.settings = replace(memory.settings, user_id="other-user")  # 改为另一个用户测试归属隔离。
    assert recovered.context("session-a") == ([], {}, {})  # 同名会话也不能跨用户读取。


def test_summary_snapshot_and_usage_roll_back_with_profile_failure(memory):  # 一轮保存必须保持事务一致。
    memory.save_turn("session-a", "原问答", "原回答", [], [change(f"字段{index}") for index in range(MAX_FACTS)])  # 先装满画像。
    before = memory.recent("session-a")  # 保存失败前的聊天记录快照。
    state = {"text": "准备保存的摘要", "through_turn_id": before[-1]["id"]}  # 新摘要与游标必须一起保存。
    with pytest.raises(ValueError, match="最多保存"):  # 容量错误应回滚整个事务。
        memory.save_turn("session-a", "不能提交的问答", "回答", [], [change("额外字段")], usage={"total": {"turns": 2}}, summary=state)  # 同时携带三种新状态。
    assert memory.recent("session-a") == before  # 失败问答、用量和摘要都不能留下半份记录。
    assert memory.context("session-a")[2] == {}  # 上轮没有摘要，失败后仍不能推进游标。
    memory.save_turn("session-a", "备用保存", "回答", [], summary=state)  # 模拟主流程回滚后只保存问答和摘要。
    assert memory.context("session-a")[2] == state  # 第二次短事务成功保存完整快照。


@pytest.mark.parametrize("state", [{"text": "", "through_turn_id": 1}, {"text": "有正文", "through_turn_id": 999}, {"text": "有正文", "through_turn_id": True}])  # 损坏状态不能静默跳过历史。
def test_context_rejects_invalid_summary_snapshot(memory, state):  # 校验覆盖位置与摘要正文必须同时有效。
    memory.save_turn("session-a", "原问题", "原回答", [], summary=state)  # 直接构造数据库中的异常快照。
    with pytest.raises(ValueError, match="摘要状态无效"):  # 读取时明确报告，而不是丢掉原文。
        memory.context("session-a")  # 使用真实 ORM 查询和状态校验。


def change(key="姓名", value="晨星", action="upsert"):
    # 造测试数据用的小工厂，让各个测试里写起来短一点。
    return FactChange(key=key, value=value, action=action, evidence="虚构原话", expires_on=None)


# ---------------------------------------------------------------------------
# 存一轮问答：字段要原样落库，而且不该顺手建出空画像
# ---------------------------------------------------------------------------
def test_save_turn_persists_original_fields_without_creating_an_empty_profile(memory):
    trace = [{"tool": "calculator", "result": "计算结果：3"}]
    # 只传四个参数，changes 用默认的 None —— 表示这一轮没有画像变化。
    memory.save_turn("session-a", "问题", "回答", trace)

    row = memory.recent("session-a")[0]

    # 读出来的字段必须和表结构完全对应（多一个少一个都说明模型定义和预期不符）。
    assert set(row) == {
        "id", "user_id", "session_id", "user_text", "assistant_text", "trace_json", "usage_json", "summary_json", "created_at",  # 摘要快照是正式持久化字段。
    }
    # id 是数据库自增的；created_at 也该被转成 datetime 对象。
    assert row["id"] > 0 and isinstance(row["created_at"], datetime)
    assert row["user_id"] == "local_user"
    assert row["user_text"] == "问题" and row["assistant_text"] == "回答"
    # 轨迹存成 JSON 文本，读回来能被解析成原来的结构。
    assert json.loads(row["trace_json"]) == trace
    assert "计算结果" in row["trace_json"]
    assert row["usage_json"] is None
    assert row["summary_json"] is None  # 未传摘要时保存 SQL NULL，不冒充已压缩。

    # 关键：这轮没有画像变化，所以不该凭空建出一条空的画像记录。
    with memory.session_factory() as session:
        assert session.get(UserMemory, "local_user") is None


# ---------------------------------------------------------------------------
# 更新画像：新值覆盖旧值，未提到的字段保持不变，来源 ID 要准
# ---------------------------------------------------------------------------
def test_orm_updates_shared_profile_and_keeps_the_actual_source_turn_id(memory):
    # 第一轮：记下姓名，顺带记一个学习目标。
    memory.save_turn("session-a", "虚构原话", "回答", [], [change(), change("目标", "学习 Python")])
    # 第二轮：在另一个会话里把姓名改成小舟。
    memory.save_turn("session-b", "虚构原话", "回答", [], [change(value="小舟")])

    profile = memory.profile()
    # 姓名被新值覆盖。
    assert profile["姓名"]["value"] == "小舟"
    # 这一轮没提“目标”，所以它必须原样保留 —— 不能因为更新别的字段就丢掉。
    assert profile["目标"]["value"] == "学习 Python"
    # 来源要指向真正提供这个值的**那一轮**（session-b 里的那一条）。
    assert profile["姓名"]["source_turn_id"] == memory.recent("session-b")[0]["id"]

    with memory.session_factory() as session:
        # 一个用户只有一行画像，更新不能变成新增。
        assert session.scalar(select(func.count()).select_from(UserMemory)) == 1
        assert isinstance(session.get(UserMemory, "local_user").updated_at, datetime)


# ---------------------------------------------------------------------------
# 遗忘：只打标记、清空值，但原始聊天一个字都不能删
# ---------------------------------------------------------------------------
def test_orm_forget_persists_json_replacement_without_deleting_history(memory):
    memory.save_turn("session-a", "虚构原话", "回答", [], [change()])
    memory.save_turn("session-b", "虚构原话", "回答", [], [change(value="", action="forget")])

    # 标记上了。
    assert memory.profile()["姓名"]["forgotten"] is True
    # 值被清空成 None（而不是直接删掉这个字段）。
    assert memory.profile()["姓名"]["value"] is None
    # 两个会话的原始聊天都还在 —— 遗忘画像不等于删除聊天。
    assert memory.recent("session-a") and memory.recent("session-b")


# ---------------------------------------------------------------------------
# 合并失败时，问答和画像要一起回滚
# ---------------------------------------------------------------------------
def test_profile_merge_failure_rolls_back_the_chat_and_keeps_old_profile(memory):
    # 先把画像填到上限（正好 MAX_FACTS 个字段）。
    memory.save_turn("session-a", "虚构原话", "回答", [], [change(f"字段{i}") for i in range(MAX_FACTS)])
    original_profile = memory.profile()

    # 再来一条就会超限，merge_profile 会抛 ValueError。
    with pytest.raises(ValueError, match="最多保存"):
        memory.save_turn("session-b", "失败问答", "回答", [], [change("超限")])

    # 关键：这一轮的聊天也没被存进去。因为两者在同一个事务里，
    # 要么一起成功，要么一起回滚，不会留下“问答存了、画像没存”的半拉子状态。
    assert memory.recent("session-b") == []
    assert memory.profile() == original_profile

    # 上一次失败不污染新的 Session，之后仍然可以只保存问答。
    memory.save_turn("session-b", "重新保存", "回答", [])
    assert len(memory.recent("session-b")) == 1


# ---------------------------------------------------------------------------
# 首次建画像就失败：连那条“空画像”也要回滚掉
# ---------------------------------------------------------------------------
def test_failed_first_profile_creation_rolls_back_the_empty_profile_too(memory, monkeypatch):
    # 直接给合并函数打补丁，让它必失败，用来模拟“首次建画像时出错”。
    monkeypatch.setattr("agent_demo.memory.merge_profile", Mock(side_effect=ValueError("测试失败")))

    with pytest.raises(ValueError, match="测试失败"):
        memory.save_turn("session-a", "虚构原话", "回答", [], [change()])

    # 问答没存住。
    assert memory.recent("session-a") == []
    # 而且那条为了合并而临时建出的空画像，也必须跟着回滚 —— 不能留下一个空壳。
    with memory.session_factory() as session:
        assert session.get(UserMemory, "local_user") is None


# ---------------------------------------------------------------------------
# 用户隔离 + 会话隔离 + 顺序 + 条数限制
# ---------------------------------------------------------------------------
def test_user_and_session_isolation_keep_existing_history_order_and_limits(memory):
    for index in range(4):
        memory.save_turn("session-a", f"问题 {index}", "回答", [])
    memory.save_turn("session-b", "另一会话", "回答", [])
    memory.save_turn("session-b", "虚构原话", "回答", [], [change()])

    # limit=2 应当拿到“最新的两条”，而且返回顺序是旧→新。
    assert [row["user_text"] for row in memory.recent("session-a", limit=2)] == ["问题 2", "问题 3"]

    # 会话列表按最近活动排序，并带上各自的轮数。
    sessions = memory.list_sessions()
    assert [(row["session_id"], row["turn_count"]) for row in sessions] == [("session-b", 2), ("session-a", 4)]
    assert len(memory.list_sessions(limit=1)) == 1

    # 把用户换成 Local_user（首字母大写）—— 因为 user_id 是大小写敏感的，
    # 这等于换了一个全新的用户，什么都该看不到。
    memory.settings = replace(memory.settings, user_id="Local_user")
    assert memory.recent("session-a") == [] and memory.list_sessions() == []
    assert memory.profile() == {}

    # 新用户自己记一条，互不干扰。
    memory.save_turn("session-a", "第二位用户", "回答", [], [change(value="小明")])
    assert memory.profile()["姓名"]["value"] == "小明"

    # 换回原用户，他的数据完好无损。
    memory.settings = replace(memory.settings, user_id="local_user")
    assert len(memory.recent("session-a")) == 4
    assert memory.profile()["姓名"]["value"] == "晨星"


# ---------------------------------------------------------------------------
# 未摘要上下文：读取全部完整问答，不再使用字数预算或固定窗口。
# ---------------------------------------------------------------------------
def test_history_context_preserves_all_complete_turns_before_summary(memory):  # 短期上下文不再使用固定窗口。
    memory.save_turn("session-a", "旧问题", "旧回答", [])
    memory.save_turn("session-a", "中问", "中答", [])
    memory.save_turn("session-a", "新问", "新答", [])

    turns, previous, summary = memory.context("session-a")  # 没有摘要时按旧到新读取所有轮次。
    assert [(turn["user_text"], turn["assistant_text"]) for turn in turns] == [("旧问题", "旧回答"), ("中问", "中答"), ("新问", "新答")]  # 每轮原文保持完整。
    assert previous == summary == {}  # 未记录用量和摘要时保持空字典语义。


# ---------------------------------------------------------------------------
# 并发首次建画像：撞主键时要靠保存点化解
# ---------------------------------------------------------------------------
def test_concurrent_first_profile_conflict_uses_savepoint_and_refreshes_locked_object(memory):
    # 用两个 get 的结果模拟另一请求已经提交画像；锁行为在真实 MySQL 中验证。
    # 这次不连数据库，直接用 Mock 冒充 Session，专门测这段异常处理逻辑。
    session = Mock()
    profile = SimpleNamespace(facts_json={})

    # 第一次 get 返回 None（以为没画像）→ 建的时候撞主键 → 第二次 get 返回别人的那行。
    session.get.side_effect = [None, profile]

    # 让 begin_nested 当上下文管理器用（模拟保存点）。
    session.begin_nested.return_value.__enter__ = Mock()
    session.begin_nested.return_value.__exit__ = Mock(return_value=False)

    # flush 时抛主键重复（1062）。error.orig.args[0] 就是那个错误码。
    session.flush.side_effect = IntegrityError("INSERT", {}, Exception(1062, "duplicate key"))

    memory._update_profile(session, [change()], 55, datetime.now().astimezone())

    # 撞了主键也不该失败：应当继续去读别人建好的画像，并把变化合并进去。
    assert profile.facts_json["姓名"]["source_turn_id"] == 55
    # 而且要带着“加行锁 + 强制读最新值”的参数去读。
    session.get.assert_called_with(UserMemory, "local_user", with_for_update=True, populate_existing=True)
    # 这里只是外层事务中的一步，不该自己提交。
    session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 别的完整性错误不能被误当成“并发建画像”
# ---------------------------------------------------------------------------
def test_other_integrity_errors_are_not_mistaken_for_concurrent_creation(memory):
    session = Mock()
    session.get.return_value = None
    session.begin_nested.return_value.__enter__ = Mock()
    session.begin_nested.return_value.__exit__ = Mock(return_value=False)

    # 这次是 1048（某列不能为空），是一个真问题，不是并发撞车。
    session.flush.side_effect = IntegrityError("INSERT", {}, Exception(1048, "null column"))

    # 所以它必须继续往外抛，不能被悄悄吞掉。
    with pytest.raises(IntegrityError):
        memory._update_profile(session, [change()], 55, datetime.now().astimezone())
