"""文档入口：校验上传原件 → 保存文件 → 提取正文 → 返回统一 Document。

只处理本地文件，不调用模型；切片和向量化由 rag.py 完成。
支持 TXT、MD、PDF、DOCX。
"""

from io import BytesIO
from pathlib import Path

# python-docx 与 LangChain 都有 Document，这里改名避免混淆。
from docx import Document as WordDocument
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_TYPES = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _read_pdf(content: bytes) -> str:
    """参数：PDF 原始字节；返回各页正文拼接的 str，不执行 OCR。"""
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise ValueError("请先解密 PDF 再上传")
    # 纯图片页可能返回 None，用空字符串保证可以拼接。
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(content: bytes) -> str:
    """参数：DOCX 原始字节；返回按正文顺序排列的段落与表格文字。"""
    document = WordDocument(BytesIO(content))
    lines = []
    # iter_inner_content() 按正文顺序同时给出段落和表格，避免表格文字丢失。
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            lines.append(block.text)
        else:
            for row in block.rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(lines)


def extract_text(name: str, content: bytes) -> str:
    """参数：带扩展名的文件名和字节；返回非空正文，失败抛 ValueError。"""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ValueError("仅支持 TXT、Markdown、PDF 和 DOCX 文件")

    try:
        if suffix in {".txt", ".md"}:
            # utf-8-sig 会吃掉 Windows 记事本可能留下的 BOM。
            text = content.decode("utf-8-sig")
        elif suffix == ".pdf":
            text = _read_pdf(content)
        else:
            text = _read_docx(content)
    # 把库的原始异常翻译成用户能看懂的提示；自家 ValueError 原样放行。
    except UnicodeDecodeError:
        raise ValueError(f"{name} 必须保存为 UTF-8 编码") from None
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"无法读取 {name}，请检查文件格式或文件是否损坏") from None

    if not text.strip():
        raise ValueError(f"{name} 没有可提取的文字；扫描 PDF 需要先做 OCR")
    return text.strip()


def save_uploads(files: list[tuple[str, bytes]], folder: Path) -> list[str]:
    """参数：文件名与字节列表、目录；返回已保存文件名，顺序与传入一致。

    先校验整批再写盘，避免坏文件导致部分上传；同名原件会覆盖。
    """
    if not files:
        raise ValueError("请先选择要上传的文档")

    # 第一遍只检查，不写盘。
    names = []
    for name, content in files:
        if not name or any(char in name for char in '\\/:<>"|?*'):
            raise ValueError("文件名不能包含路径或特殊字符")
        if name in names:
            raise ValueError("同一批文件中有重名，请修改文件名后上传")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{name} 超过单文件 10 MB 限制")
        extract_text(name, content)  # 试拆一次，确认文件真的能读。
        names.append(name)

    folder.mkdir(parents=True, exist_ok=True)
    for name, content in files:
        target = (folder / name).resolve()
        # 第二道路径穿越检查：最终落点必须直接位于上传目录。
        if target.parent != folder.resolve():
            raise ValueError("上传文件必须保存在上传目录内")
        target.write_bytes(content)
    return names


def load_documents(docs_dir: Path, uploads_dir: Path) -> list[Document]:
    """参数：预置与上传目录；返回含正文和来源的 Document 列表。

    上传来源加 uploads/ 前缀；sorted 保证切片编号稳定。
    """
    documents = []
    folders = [(docs_dir, ""), (uploads_dir, "uploads/")]

    for folder, prefix in folders:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_TYPES:
                continue
            text = extract_text(path.name, path.read_bytes())
            source = prefix + path.relative_to(folder).as_posix()
            documents.append(Document(page_content=text, metadata={"source": source}))

    return documents
