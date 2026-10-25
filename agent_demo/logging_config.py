"""日志开关：用 Python 自带的 logging，把运行信息和报错写到本地。"""

import logging
from logging.handlers import RotatingFileHandler

from agent_demo.config import ROOT


LOG_FILE = ROOT / "data" / "logs" / "app.log"


def setup_logging() -> None:
    """入口调用一次；后续页面重跑直接复用，避免一条日志被重复写多遍。"""
    logger = logging.getLogger("agent_demo")
    if logger.handlers:
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 写满 5 MB 就换新文件；旧文件依次是 app.log.1、.2、.3，最多留三份。
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    # 同一份日志也写到 PyCharm 控制台，不必每次都打开文件。
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s")
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    # 本项目的日志到这里就结束，避免再被 Streamlit 的日志配置打印一次。
    logger.propagate = False
