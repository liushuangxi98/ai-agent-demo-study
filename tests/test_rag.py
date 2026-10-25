"""知识库的本地测试：不请求模型，用内存书架检查证据格式与同步顺序。"""

import threading
import os
from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import agent_demo.rag as rag_module
from agent_demo.rag import KnowledgeBase


@pytest.mark.skipif(os.name != "nt", reason="只验证 Windows 临时目录修复")
def test_windows_chroma_uses_local_temp_and_works_without_network(tmp_path, monkeypatch, settings):
    class LocalEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    # 故意给不存在的默认临时目录；构造时必须改为项目自身的可写目录。
    monkeypatch.setenv("TEMP", str(tmp_path / "missing-temp"))
    monkeypatch.setenv("TMP", str(tmp_path / "missing-temp"))
    monkeypatch.setattr(rag_module, "build_embeddings", lambda settings: LocalEmbeddings())
    monkeypatch.setattr(httpx.Client, "send", Mock(side_effect=AssertionError("测试禁止联网")))
    local_settings = replace(settings, data_dir=tmp_path / "data")
    knowledge = KnowledgeBase(local_settings)  # 不执行 ingest，不读取任何真实文档。
    try:
        assert os.environ["TEMP"] == os.environ["TMP"] == str((local_settings.data_dir / "tmp").resolve())
        assert knowledge.store.get()["ids"] == []
        knowledge.store.add_documents([Document(page_content="虚构测试资料")], ids=["test-chunk"])
        assert knowledge.store.similarity_search("虚构问题", k=1)[0].page_content == "虚构测试资料"
    finally:
        knowledge.store._client._system.stop()


def test_context_formats_sources_without_changing_the_evidence():
    """合并排版方法后，来源编号、换行和片段顺序仍然一致。"""
    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.retrieve = Mock(return_value=[
        {"source": "course.md", "chunk": 0, "text": "课程说明。"},
        {"source": "uploads/notes.txt", "chunk": 2, "text": "上传资料。"},
    ])

    assert knowledge.context("课程") == (
        "[course.md#0]\n课程说明。\n\n[uploads/notes.txt#2]\n上传资料。"
    )
    knowledge.retrieve.assert_called_once_with("课程")


def test_concurrent_sync_does_not_restore_an_older_file_snapshot():
    """第一次同步暂停时，第二次不能抢先读原件再更新；最终应保留新内容。"""
    first_prepared = threading.Event()
    release_first = threading.Event()
    second_waiting = threading.Event()
    second_prepared = threading.Event()
    errors = []

    class ObservedLock:
        """在第二个线程准备拿锁时通知测试，不靠 sleep 猜它走到哪里了。"""

        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.current_thread().name == "second-sync":
                second_waiting.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.lock = ObservedLock()
    stored_ids = {"old"}
    current_file = ["old"]
    knowledge.store = Mock()
    knowledge.store.get.side_effect = lambda: {"ids": list(stored_ids)}
    knowledge.store.add_documents.side_effect = lambda chunks, ids: stored_ids.update(ids)
    knowledge.store.delete.side_effect = lambda ids: stored_ids.difference_update(ids)

    def prepare():
        snapshot = current_file.copy()
        if threading.current_thread().name == "first-sync":
            first_prepared.set()
            if not release_first.wait(5):
                raise TimeoutError("测试没有释放第一次同步")
        else:
            second_prepared.set()
        return snapshot, snapshot

    knowledge._prepare_chunks = prepare

    def sync():
        try:
            knowledge.ingest()
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=sync, name="first-sync", daemon=True)
    second = threading.Thread(target=sync, name="second-sync", daemon=True)
    first.start()
    try:
        assert first_prepared.wait(5)
        current_file[:] = ["new"]
        second.start()
        assert second_waiting.wait(5)
        assert not second_prepared.is_set()
    finally:
        release_first.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert second_prepared.is_set()
    assert stored_ids == {"new"}
