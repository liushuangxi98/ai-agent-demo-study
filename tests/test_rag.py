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
    knowledge.is_empty = Mock(return_value=False)
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


def test_ingest_empty_directories_clears_index_and_returns_zero():
    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.lock = threading.RLock()
    knowledge.store = Mock()
    knowledge.store.get.return_value = {"ids": ["old-1", "old-2"]}
    knowledge._prepare_chunks = Mock(return_value=([], []))

    assert knowledge.ingest() == 0
    assert set(knowledge.store.delete.call_args.kwargs["ids"]) == {"old-1", "old-2"}
    knowledge.store.add_documents.assert_not_called()


def test_empty_knowledge_does_not_search_and_returns_message():
    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.lock = threading.RLock()
    knowledge.store = Mock()
    knowledge.store.get.return_value = {"ids": []}

    assert knowledge.retrieve("问题") == []
    assert "知识库为空" in knowledge.context("问题")
    knowledge.store.similarity_search_with_score.assert_not_called()


def test_delete_removes_docs_and_uploads_and_syncs(tmp_path, settings):
    docs, uploads = tmp_path / "docs", tmp_path / "uploads"
    docs.mkdir()
    uploads.mkdir()
    (docs / "course.md").write_text("课程资料", encoding="utf-8")
    (uploads / "notes.txt").write_text("上传资料", encoding="utf-8")

    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.lock = threading.RLock()
    knowledge.settings = replace(settings, docs_dir=docs)
    knowledge.uploads_dir = uploads
    knowledge.ingest = Mock(return_value=2)

    assert "已删除 course.md" in knowledge.delete("course.md")
    assert not (docs / "course.md").exists()
    assert "已删除 uploads/notes.txt" in knowledge.delete("uploads/notes.txt")
    assert not (uploads / "notes.txt").exists()
    assert knowledge.ingest.call_count == 2


def test_delete_rejects_outside_paths_and_missing_files(tmp_path, settings):
    docs, uploads = tmp_path / "docs", tmp_path / "uploads"
    docs.mkdir()
    uploads.mkdir()

    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.lock = threading.RLock()
    knowledge.settings = replace(settings, docs_dir=docs)
    knowledge.uploads_dir = uploads
    knowledge.ingest = Mock()

    assert "来源名" in knowledge.delete("../.env")
    assert "来源名" in knowledge.delete(str(tmp_path / "outside.txt"))
    assert "来源名" in knowledge.delete("uploads/../outside.txt")
    assert "未找到" in knowledge.delete("missing.md")
    knowledge.ingest.assert_not_called()


def test_delete_last_document_reports_empty_library(tmp_path, settings):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "course.md").write_text("课程资料", encoding="utf-8")

    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    knowledge.lock = threading.RLock()
    knowledge.settings = replace(settings, docs_dir=docs)
    knowledge.uploads_dir = tmp_path / "uploads"
    knowledge.ingest = Mock(return_value=0)

    assert "知识库已清空" in knowledge.delete("course.md")
