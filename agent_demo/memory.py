"""MySQL 记忆：用 SQLAlchemy ORM 保存聊天和跨会话共享画像。

ConversationTurn 一行对应一轮问答，UserMemory 一行对应一个用户画像。
业务读写交给 Session；建库、建表属于结构初始化。
"""

import json
import re
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, create_engine, func, select
from sqlalchemy.dialects.mysql import JSON, LONGTEXT
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.schema import CreateSchema

from agent_demo.config import Settings
from agent_demo.structured_memory import FactChange, merge_profile


class Base(DeclarativeBase):
    """ORM 模型基类；Base.metadata 收集表结构。"""


class ConversationTurn(Base):
    """一轮聊天，对应 conversation_turns 表；id、created_at 由数据库生成。"""

    __tablename__ = "conversation_turns"
    __table_args__ = (
        Index("ix_user_session_id", "user_id", "session_id", "id"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # user_id 使用字节比较，保证 Local_user 与 local_user 是两个用户。
    user_id: Mapped[str] = mapped_column(String(64, collation="utf8mb4_bin"))
    session_id: Mapped[str] = mapped_column(String(64))
    user_text: Mapped[str] = mapped_column(Text)
    assistant_text: Mapped[str] = mapped_column(Text)
    trace_json: Mapped[str] = mapped_column(LONGTEXT)
    # none_as_null=True：没有用量时存 NULL，不冒充 0。
    usage_json: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    def to_dict(self) -> dict[str, Any]:
        """返回普通字典，避免 Session 关闭后再访问 ORM 对象。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "trace_json": self.trace_json,
            "usage_json": self.usage_json,
            "summary_json": self.summary_json,
            "created_at": self.created_at,
        }


class UserMemory(Base):
    """用户画像，对应 user_memories 表；user_id 是主键，一个用户只有一行。"""

    __tablename__ = "user_memories"
    __table_args__ = {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}

    user_id: Mapped[str] = mapped_column(String(64, collation="utf8mb4_bin"), primary_key=True)
    facts_json: Mapped[dict[str, dict[str, Any]]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


def validate_session_id(session_id: str) -> str:
    """参数：待校验的会话编号；格式合法返回原值，否则抛 ValueError。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        raise ValueError("会话 ID 只能包含 1～64 个字母、数字、下划线或连字符")
    return session_id


class ConversationMemory:
    """管理记忆读写；每次操作创建独立 Session，用完即关。"""

    def __init__(self, settings: Settings) -> None:
        """参数：settings 提供 MySQL 配置和用户 ID；建库建表后返回 None。"""
        self.settings = settings
        database = settings.mysql_database

        # URL.create 会正确处理密码里的 @ 等特殊字符。
        server_url = URL.create(
            "mysql+pymysql",
            username=settings.mysql_user,
            password=settings.mysql_password,
            host=settings.mysql_host,
            port=settings.mysql_port,
            query={"charset": "utf8mb4"},
        )
        # pool_pre_ping：取连接前先探活；connect_timeout：连不上最多等 5 秒。
        options = {"pool_pre_ping": True, "connect_args": {"connect_timeout": 5}}

        # 目标库可能还不存在，因此先连服务、不指定库名。
        server = create_engine(server_url, **options)
        try:
            with Session(server) as session:
                session.execute(CreateSchema(database, if_not_exists=True))
                session.commit()
        finally:
            server.dispose()

        self.engine = create_engine(server_url.set(database=database), **options)
        try:
            # create_all 只创建缺失表，不修改已有表结构。
            Base.metadata.create_all(self.engine)
            self.session_factory = sessionmaker(bind=self.engine)
        except Exception:
            self.engine.dispose()
            raise

    def profile(self) -> dict[str, dict[str, Any]]:
        """无额外参数；返回当前用户共享画像，没有记录时返回空字典。"""
        with self.session_factory() as session:
            profile = session.get(UserMemory, self.settings.user_id)
            return profile.facts_json if profile is not None else {}

    def save_turn(
        self, session_id: str, question: str, answer: str,
        trace: list[dict[str, Any]], changes: list[FactChange] | None = None,
        now: datetime | None = None,
        *, usage: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """在一个事务中保存问答与画像变化，返回 None。

        参数：会话 ID、用户原话、最终回答、工具/反思轨迹、画像变化、记录时间，
        以及关键字参数 usage（本轮与会话累计）和 summary（摘要快照）。
        session_id 已由 Agent 入口校验；失败抛异常并整体回滚。
        """
        now = now or datetime.now().astimezone()

        turn = ConversationTurn(
            user_id=self.settings.user_id,
            session_id=session_id,
            user_text=question,
            assistant_text=answer,
            trace_json=json.dumps(trace, ensure_ascii=False),
            usage_json=usage,
            summary_json=summary,
        )

        # begin() 正常结束提交、抛异常回滚；问答与画像要么一起成功，要么一起不算数。
        with self.session_factory.begin() as session:
            session.add(turn)

            if changes:
                # 先 flush 取得自增 id，画像的 source_turn_id 才能指向本轮问答。
                session.flush()
                self._update_profile(session, changes, turn.id, now)

    def _update_profile(
        self, session: Session, changes: list[FactChange],
        turn_id: int, now: datetime,
    ) -> None:
        """参数：当前 Session、变化列表、来源问答 ID、时间；返回 None。

        复用 save_turn 已开的事务，不单独提交，由外层统一决定成败。
        """
        user_id = self.settings.user_id

        if session.get(UserMemory, user_id) is None:
            try:
                # 保存点只保护“创建空画像”；两个请求同时首次建画像时，
                # 后来的请求撞主键 1062，只需撤销这一步，再读取已有画像。
                with session.begin_nested():
                    session.add(UserMemory(user_id=user_id, facts_json={}))
                    session.flush()
            except IntegrityError as error:
                if error.orig.args[0] != 1062:
                    raise

        # 行锁 + 强制读最新值，避免并发更新互相覆盖。
        profile = session.get(UserMemory, user_id, with_for_update=True, populate_existing=True)
        # 前面已确保画像行存在；整体换新字典，ORM 才能可靠识别 JSON 列变化。
        profile.facts_json = merge_profile(profile.facts_json, changes, turn_id, now)
        profile.updated_at = func.current_timestamp()  # 更新时间取 MySQL 服务端时钟。

    def recent(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """参数：会话 ID、最多读取轮数；返回按旧到新排列的问答字典列表。"""
        session_id = validate_session_id(session_id)

        statement = (
            select(ConversationTurn)
            .where(
                ConversationTurn.user_id == self.settings.user_id,
                ConversationTurn.session_id == session_id,
            )
            .order_by(ConversationTurn.id.desc())  # 先取最新，再翻回正序。
            .limit(limit)
        )

        with self.session_factory() as session:
            turns = session.scalars(statement).all()
            return [turn.to_dict() for turn in reversed(turns)]

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """参数：最多会话数；返回当前用户的会话列表，按最近活动倒序。"""
        # 没有会话表，直接从聊天记录分组统计；空会话自然不会出现。
        statement = (
            select(
                ConversationTurn.session_id,
                func.count().label("turn_count"),
                func.max(ConversationTurn.created_at).label("last_saved_at"),
            )
            .where(ConversationTurn.user_id == self.settings.user_id)
            .group_by(ConversationTurn.session_id)
            .order_by(func.max(ConversationTurn.id).desc())
            .limit(limit)
        )

        with self.session_factory() as session:
            rows = session.execute(statement).all()
            return [
                {"session_id": row.session_id, "turn_count": row.turn_count,
                 "last_saved_at": row.last_saved_at}
                for row in rows
            ]

    def context(self, session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """参数：当前会话 ID（已由 Agent 入口校验）；返回摘要后问答、上轮累计、摘要状态。"""
        latest_query = (
            select(ConversationTurn)
            .where(
                ConversationTurn.user_id == self.settings.user_id,
                ConversationTurn.session_id == session_id,
            )
            .order_by(ConversationTurn.id.desc())
            .limit(1)  # 直接读取最后一行，累计和摘要都取它的快照。
        )

        with self.session_factory() as session:
            latest = session.scalar(latest_query)
            if latest is None:
                return [], {}, {}

            previous_usage = (latest.usage_json or {}).get("total", {})
            summary_state = latest.summary_json or {}
            through_turn_id = summary_state.get("through_turn_id", 0) if isinstance(summary_state, dict) else 0
            # 有摘要时，正文和游标必须同时有效，不能用损坏快照悄悄跳过原文。
            if not isinstance(summary_state, dict) or (
                summary_state and (
                    not isinstance(summary_state.get("text"), str)
                    or not summary_state["text"].strip()
                    or type(through_turn_id) is not int
                    or not 0 < through_turn_id <= latest.id
                )
            ):
                raise ValueError("历史摘要状态无效，请检查摘要正文与覆盖位置")

            turns_query = (
                select(ConversationTurn)
                .where(
                    ConversationTurn.user_id == self.settings.user_id,
                    ConversationTurn.session_id == session_id,
                    ConversationTurn.id > through_turn_id,
                    ConversationTurn.id <= latest.id,  # 与本次读取的最新快照保持一致。
                )
                .order_by(ConversationTurn.id)  # 模型按旧到新阅读。
            )
            turns = [turn.to_dict() for turn in session.scalars(turns_query)]
            return turns, previous_usage, summary_state

    def close(self) -> None:
        """无额外参数；释放连接池，不删除数据。"""
        self.engine.dispose()
