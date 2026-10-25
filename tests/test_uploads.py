"""检查四种文档的正文提取和上传校验；不创建额外应用运行模式。"""

# ---------------------------------------------------------------------------
# 标准库导入
# ---------------------------------------------------------------------------
# BytesIO 把一段字节包装成“像文件一样”的对象，
# 因为生成 PDF / DOCX 的库都要求写给一个文件对象，而我们在内存里造测试数据。
from io import BytesIO

# ---------------------------------------------------------------------------
# 第三方库导入
# ---------------------------------------------------------------------------
# pytest 是测试框架。核心用法就三个：
#   @pytest.mark.parametrize  让一个测试函数跑多组数据
#   pytest.raises(...)        断言“这段代码应该抛异常”
#   tmp_path                  一个内置的临时目录，测试结束自动清理
import pytest

# python-docx：用来在内存里造一份真的 DOCX 文件当测试数据。
from docx import Document

# pypdf：用来在内存里造一份真的 PDF。
# 前几个是高层 API；后面 generic 里的是低层对象，手工拼页面内容时要用到。
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from agent_demo.documents import MAX_UPLOAD_BYTES, extract_text, save_uploads


# ---------------------------------------------------------------------------
# 测试数据工厂：造一份“有文字”的 PDF
# ---------------------------------------------------------------------------
def make_pdf():
    # 这几行是在手工拼一份最小的 PDF：新建文档 → 加一页 → 放一个字体资源
    # → 往页面里写一条文字绘制指令。看着琐碎，但好处是不用外挂一个二进制测试文件。
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)

    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })

    # 这一串是 PDF 内部的绘制语法：设置字体、移动到坐标、输出文字。
    # 所以我们这张“图”里其实是有真实文字层的，能被解析出来。
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Orion secret is blue pinecone.) Tj ET")
    page[NameObject("/Contents")] = stream

    result = BytesIO()
    writer.write(result)
    # getvalue() 把内存里攒好的字节取出来，交给被测函数使用。
    return result.getvalue()


# ---------------------------------------------------------------------------
# 测试数据工厂：造一份“有正文也有表格”的 DOCX
# ---------------------------------------------------------------------------
def make_docx():
    document = Document()
    document.add_paragraph("极光计划口令：蓝色松果。")

    # 特意加一张表格，是为了验证“表格里的文字也能被抠出来”。
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "有效期"
    table.cell(0, 1).text = "30 天"

    result = BytesIO()
    document.save(result)
    return result.getvalue()


# ---------------------------------------------------------------------------
# 四种格式都能上传并提取出文字
# ---------------------------------------------------------------------------
# @pytest.mark.parametrize 是装饰器：把同一个测试函数用多组参数各跑一遍。
# 这里等于一口气测了 txt / md / pdf / docx 四次，报错时也会分别指出是哪一种。
@pytest.mark.parametrize("suffix", [".txt", ".md", ".pdf", ".docx"])
def test_document_upload(tmp_path, suffix):
    # tmp_path 是 pytest 自动提供的临时目录：每个测试一个独立目录，跑完自动清理。
    content = "极光计划口令：蓝色松果。".encode("utf-8-sig")
    expected = "蓝色松果"

    if suffix == ".pdf":
        # 换成真的 PDF，并改判“能提取到英文那句”。
        content, expected = make_pdf(), "blue pinecone"
    if suffix == ".docx":
        content = make_docx()

    name = "notes" + suffix

    # 保存成功，并且返回的确实是这份文件名。
    assert save_uploads([(name, content)], tmp_path) == [name]
    # 落盘的内容和原来给的字节完全一致，没有中途改动。
    assert (tmp_path / name).read_bytes() == content
    # 能从正文里抠出关键内容。
    assert expected in extract_text(name, content)

    # DOCX 额外验证一次表格：说明表格里的格子文字也被提取到了。
    if suffix == ".docx":
        assert "30 天" in extract_text(name, content)


# ---------------------------------------------------------------------------
# 坏文件必须被拦住，而且不能留下“半拉子”结果
# ---------------------------------------------------------------------------
# 这个参数表里全是应当被拒绝的输入：
#   ../notes.txt  和 ..\notes.txt  → 试图用相对路径把文件写到别的目录（路径穿越）
#   notes.csv                     → 不支持的格式
#   b""                           → 空文件
#   b"\xff"                       → 不是合法 UTF-8 编码
#   b"not a PDF"                  → 内容是假的 PDF
@pytest.mark.parametrize("name, content", [
    ("../notes.txt", b"text"), ("..\\notes.txt", b"text"),
    ("notes.csv", b"text"), ("notes.txt", b""),
    ("notes.txt", b"\xff"), ("notes.pdf", b"not a PDF"),
])
def test_bad_upload_does_not_save_partial_batch(tmp_path, name, content):
    # 一批文件里故意混一个坏的，前面那个是好的。
    # 期望：整体报错，而且“好的那个”也不许被写进磁盘。
    # 这正是 save_uploads “先全部检查、再一起写盘”的设计意图 —— 要么全成，要么全不写。
    with pytest.raises(ValueError):
        save_uploads([("good.txt", b"valid text"), (name, content)], tmp_path / "uploads")

    assert not (tmp_path / "uploads" / "good.txt").exists()


# ---------------------------------------------------------------------------
# 大小与空批次的边界
# ---------------------------------------------------------------------------
def test_upload_limits(tmp_path):
    # 超大文件要报错，且提示里应该包含 "10 MB" 字样。
    with pytest.raises(ValueError, match="10 MB"):
        save_uploads([("large.txt", b"a" * (MAX_UPLOAD_BYTES + 1))], tmp_path)

    # 一个文件都没选也要报错。
    with pytest.raises(ValueError, match="先选择"):
        save_uploads([], tmp_path)


# ---------------------------------------------------------------------------
# 扫描版 PDF（有页面但没有文字层）要给出可操作的提示
# ---------------------------------------------------------------------------
def test_pdf_without_text_is_rejected():
    # 空白页 = 只有图形、没有文字，正是“扫描件”的典型情况。
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    result = BytesIO()
    writer.write(result)

    # 提示里要出现 "OCR"，直接告诉用户“这种文件得先做文字识别”。
    with pytest.raises(ValueError, match="OCR"):
        extract_text("scan.pdf", result.getvalue())
