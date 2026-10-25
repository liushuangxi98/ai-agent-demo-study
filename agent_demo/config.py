"""读取统一配置：.env / 环境变量 → 类型转换 → Settings。

创建 Settings 时自动完成范围、必填项、地址和端口检查。
本模块不连接模型或数据库。其他模块只接收 Settings，不各自读取 .env。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# 从本文件位置往上两级定位项目根目录，避免受启动目录影响。
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """应用配置；创建后不可修改，评测用 dataclasses.replace 复制后改字段。"""

    # repr=False：打印配置对象时隐藏密钥和密码。
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = ""
    llm_model: str = "qwen3.8-flash"
    embedding_model: str = "text-embedding-v4"

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = field(default="", repr=False)
    mysql_database: str = "agent_demo"

    user_id: str = "local_user"  # 数据归属标签，不是登录认证。
    context_token_budget: int = 50000  # 主问答最后一次返回的输入加输出严格超过此值才摘要。
    summary_keep_turns: int = 6  # 压缩时保留最近几轮完整问答；之后原文继续增长。
    rag_top_k: int = 3
    rag_max_distance: float = 0.8  # 余弦距离上限，越小越严格。
    reflection_enabled: bool = False  # 默认关闭，避免每轮多一次模型请求。

    data_dir: Path = ROOT / "data"
    docs_dir: Path = ROOT / "docs" / "knowledge"

    def __post_init__(self) -> None:
        """字段赋值后自动检查配置；失败抛 ValueError。"""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.user_id):
            raise ValueError("USER_ID 只能包含 1～64 个字母、数字、下划线或连字符")

        if min(self.context_token_budget, self.summary_keep_turns, self.rag_top_k) < 1:
            raise ValueError("摘要阈值、原文保留轮数和检索数量必须大于 0")

        if not 0 <= self.rag_max_distance <= 2:
            raise ValueError("RAG_MAX_DISTANCE 应在 0～2 之间")

        self.validate()

    def validate(self) -> None:
        """由 __post_init__ 调用，检查必填项和连接参数；失败抛 ValueError。"""
        required = {
            "LLM_API_KEY": self.llm_api_key,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL": self.llm_model,
            "EMBEDDING_MODEL": self.embedding_model,
            "MYSQL_USER": self.mysql_user,
            "MYSQL_PASSWORD": self.mysql_password,
            "MYSQL_DATABASE": self.mysql_database,
        }
        # 只收集配置名，不把密钥或密码放进错误提示。
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError("请在 .env 中填写：" + "、".join(missing))

        url = urlparse(self.llm_base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("LLM_BASE_URL 必须是有效的 API 地址")

        if not 1 <= self.mysql_port <= 65535:
            raise ValueError("MYSQL_PORT 必须在 1～65535 之间")


def load_settings() -> Settings:
    """无参数；读取配置并返回 Settings，系统环境变量优先。"""
    # override=False：系统里已有同名环境变量时，以系统为准。
    load_dotenv(ROOT / ".env", override=False, encoding="utf-8")
    env = os.environ

    # 环境变量都是字符串，数字和布尔值在这里转换。
    return Settings(
        llm_api_key=env.get("LLM_API_KEY", ""),
        llm_base_url=env.get("LLM_BASE_URL", ""),
        llm_model=env.get("LLM_MODEL", "qwen3.8-flash"),
        embedding_model=env.get("EMBEDDING_MODEL", "text-embedding-v4"),
        mysql_host=env.get("MYSQL_HOST", "127.0.0.1"),
        mysql_port=int(env.get("MYSQL_PORT", "3306")),
        mysql_user=env.get("MYSQL_USER", ""),
        mysql_password=env.get("MYSQL_PASSWORD", ""),
        mysql_database=env.get("MYSQL_DATABASE", "agent_demo"),
        user_id=env.get("USER_ID", "local_user"),
        context_token_budget=int(env.get("CONTEXT_TOKEN_BUDGET", "50000")),
        summary_keep_turns=int(env.get("SUMMARY_KEEP_TURNS", "6")),
        rag_top_k=int(env.get("RAG_TOP_K", "3")),
        rag_max_distance=float(env.get("RAG_MAX_DISTANCE", "0.8")),
        # 不能直接 bool("false")，那样会得到 True。
        reflection_enabled=env.get("REFLECTION_ENABLED", "false").strip().lower() in {"true", "1", "yes", "on"},
    )
