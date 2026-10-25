"""只用虚构异常验证落盘、完整堆栈、文件轮转和网页启动失败提示。"""

import logging

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import agent_demo.agent as agent_module
import agent_demo.config as config_module
import agent_demo.logging_config as log_module


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    logger = logging.getLogger("agent_demo")
    old_handlers, old_level, old_propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    path = tmp_path / "logs" / "app.log"
    monkeypatch.setattr(log_module, "LOG_FILE", path)
    try:
        yield path, logger
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_logging_keeps_utf8_traceback_and_does_not_duplicate_handlers(isolated_log):
    path, logger = isolated_log
    log_module.setup_logging()
    log_module.setup_logging()  # 模拟页面重跑，日志不能因此写两遍。
    child = logging.getLogger("agent_demo.test")
    child.info("测试初始化成功")
    try:
        raise RuntimeError("虚构的数据库打开失败")
    except RuntimeError:
        child.exception("测试页面操作失败")

    content = path.read_text(encoding="utf-8")
    assert content.count("测试页面操作失败") == 1
    assert "INFO agent_demo.test:" in content and "测试初始化成功" in content
    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: 虚构的数据库打开失败" in content
    assert "test_logging_keeps_utf8_traceback" in content
    assert len(logger.handlers) == 2


def test_logs_rotate_without_growing_without_limit(isolated_log):
    path, logger = isolated_log
    log_module.setup_logging()
    file_handler = next(handler for handler in logger.handlers if isinstance(handler, logging.FileHandler))
    assert file_handler.maxBytes == 5 * 1024 * 1024 and file_handler.backupCount == 3
    file_handler.maxBytes = 200  # 测试中缩小阈值，避免真的写满五兆。
    for number in range(20):
        logger.info("轮转测试第 %s 条 %s", number, "x" * 80)
    assert path.with_name("app.log.3").exists()
    assert len(list(path.parent.glob("app.log*"))) == 4
    assert "第 19 条" in path.read_text(encoding="utf-8")


def test_startup_error_is_saved_while_page_shows_only_short_message(isolated_log, monkeypatch, settings):
    path, _ = isolated_log
    def fail(settings):
        raise RuntimeError("虚构的底层故障详情")
    monkeypatch.setattr(agent_module, "AgentApp", fail)
    monkeypatch.setattr(config_module, "load_settings", lambda: settings)
    st.cache_resource.clear()
    try:
        page = AppTest.from_file(config_module.ROOT / "app.py").run()
        assert not page.exception
        assert any("RuntimeError" in item.value for item in page.error)
        assert all("底层故障详情" not in item.value for item in page.error)
        assert any(str(path) in item.value for item in page.caption)
        content = path.read_text(encoding="utf-8")
        assert "页面操作失败" in content and "RuntimeError: 虚构的底层故障详情" in content
        assert "get_app" in content
    finally:
        st.cache_resource.clear()
