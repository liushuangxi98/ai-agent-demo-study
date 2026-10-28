"""知识库：原件 → 切片 → 增量向量化入库；问题 → 检索 → 带来源的证据。

本模块不生成最终回答，证据由 agent.py 交给模型。
"""

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_demo.config import Settings
from agent_demo.documents import load_documents, save_uploads
from agent_demo.embeddings import build_embeddings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


class KnowledgeBase:
    """管理共享知识；settings 是配置，store 是 Chroma，uploads_dir 存原件。"""

    def __init__(self, settings: Settings) -> None:
        """参数：settings 提供模型、目录与检索参数；返回 None。"""
        self.settings = settings
        self.uploads_dir = settings.data_dir / "uploads"

        # Windows 下 Chroma 需要可写的临时目录，默认 TEMP 异常会报“无法打开数据库”。
        if os.name == "nt":
            temp_dir = settings.data_dir / "tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TEMP"] = str(temp_dir.resolve())
            os.environ["TMP"] = str(temp_dir.resolve())

        # upload() 内部会再调 ingest()，因此用可重入锁。
        self.lock = threading.RLock()

        # 集合名带模型、接口和切片规则的版本号；任一变化就换新集合，避免旧向量混用。
        identity = (
            f"{settings.embedding_model}:{settings.llm_base_url}:"
            f"split-{CHUNK_SIZE}-{CHUNK_OVERLAP}"
        )
        version = hashlib.sha256(identity.encode()).hexdigest()[:12]

        self.store = Chroma(
            collection_name=f"learning_{version}",
            embedding_function=build_embeddings(settings),
            persist_directory=str(settings.data_dir / "chroma"),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _prepare_chunks(self) -> tuple[list[Document], list[str]]:
        """无额外参数；返回（切片列表, ID 列表），两者按下标一一对应。

        只做本地读取、切片和指纹计算，不调用向量接口。
        """
        documents = load_documents(self.settings.docs_dir, self.uploads_dir)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", " ", ""],
        )

        chunks, ids = [], []
        for document in documents:
            # 每份文档单独编号，新增文件不会让其他文件的切片编号漂移。
            for number, chunk in enumerate(splitter.split_documents([document])):
                chunk.metadata["chunk"] = number
                chunks.append(chunk)
                # 指纹包含来源、编号和正文；内容不变就能在下次同步时跳过向量化。
                identity = f"{chunk.metadata['source']}:{number}:{chunk.page_content}"
                ids.append(hashlib.sha256(identity.encode()).hexdigest())

        if not chunks:
            return [], []
        return chunks, ids

    def ingest(self) -> int:
        """无额外参数；同步当前原件，返回全部切片数（不是本次新增数）。"""
        with self.lock:
            chunks, ids = self._prepare_chunks()
            old_ids = set(self.store.get()["ids"])

            # 目录里一份文档都没有时，清空旧索引；空库是允许的状态。
            if not chunks:
                if old_ids:
                    self.store.delete(ids=list(old_ids))
                return 0

            new_chunks, new_ids = [], []
            for chunk, chunk_id in zip(chunks, ids):
                if chunk_id not in old_ids:
                    new_chunks.append(chunk)
                    new_ids.append(chunk_id)

            # 先添加、后删除：向量化失败时旧知识仍然可用。
            if new_chunks:
                self.store.add_documents(new_chunks, ids=new_ids)
            obsolete_ids = old_ids - set(ids)
            if obsolete_ids:
                self.store.delete(ids=list(obsolete_ids))

        return len(chunks)

    def upload(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """参数：文件名与字节列表；返回 {"files": 已保存文件名, "chunks": 切片总数}。"""
        with self.lock:
            names = save_uploads(files, self.uploads_dir)
            # 向量化失败后原件仍保留，可再次同步。
            count = self.ingest()
        return {"files": names, "chunks": count}

    def is_empty(self) -> bool:
        """无额外参数；知识库没有任何切片时返回 True。"""
        with self.lock:
            return not self.store.get(limit=1)["ids"]

    def delete(self, source: str) -> str:
        """参数：知识来源名；删除原件并同步索引，返回结果文字。"""
        with self.lock:
            raw = source.strip()
            if not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
                return "source 必须是知识文档来源名，例如 course.md 或 uploads/notes.txt"
            if raw.startswith("uploads/"):
                root, relative = self.uploads_dir, raw.removeprefix("uploads/")
            else:
                root, relative = self.settings.docs_dir, raw

            target = (root / relative).resolve()
            if root.resolve() not in target.parents:
                return "source 必须位于知识目录内"
            if not target.is_file():
                return f"未找到知识文档：{raw}"

            try:
                target.unlink()
            except OSError as error:
                return f"删除失败：{error}"

            try:
                count = self.ingest()
            except Exception as error:
                logger.exception("删除后同步知识库失败：%s", raw)
                return f"文件已删除，但知识库同步失败：{type(error).__name__}"

        return f"已删除 {raw}，知识库现有 {count} 个切片。" if count else f"已删除 {raw}，知识库已清空。"

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """参数：问题或检索词；返回命中列表，每项含 source、chunk、distance、text。"""
        if self.is_empty():
            return []
        with self.lock:
            matches = self.store.similarity_search_with_score(query, k=self.settings.rag_top_k)

        hits = []
        for document, distance in matches:
            # 距离阈值只筛“资料相关性”，不是答案正确率。
            if distance <= self.settings.rag_max_distance:
                hits.append({
                    "source": document.metadata["source"],
                    "chunk": document.metadata["chunk"],
                    "distance": round(float(distance), 4),
                    "text": document.page_content,
                })
        return hits

    def context(self, query: str) -> str:
        """参数：检索问题；返回带来源的证据，未命中或空库时返回一句提示。"""
        if self.is_empty():
            return "知识库为空，请先上传或同步文档。"
        hits = self.retrieve(query)
        if not hits:
            # 未命中不抛错：模型仍可回答常识或使用其他工具。
            return "知识库未找到足够相关的证据。"
        passages = [f"[{hit['source']}#{hit['chunk']}]\n{hit['text']}" for hit in hits]
        return "\n\n".join(passages)
