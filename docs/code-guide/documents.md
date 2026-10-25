# documents.py：仓库前台——收件、上架、抠文字

对应源码：[agent_demo/documents.py](D:/study/AI/ai-agent-demo-study/agent_demo/documents.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

知识库流水线的**第一站**。你的 PDF/Word/TXT 不能直接进向量库，得先有人：**收件验货（能不能收）、上架保存（存原件）、拆包取字（抠出纯文字）**，最后打包成统一格式的"文字包裹"交给 `rag.py`。

这个文件**不碰 AI**：不调模型、不算向量、不写 MySQL。纯粹的"文件搬运 + 拆包工"。

## 0. 先分清三种"东西"

后面所有代码都绕着这三个转：

| 名字 | 白话 | 例子 |
| --- | --- | --- |
| `bytes`（字节）| 文件在硬盘上的原始二进制，什么都能装（文字、图片、格式信息）| 网页上传递来的 `content` |
| `str`（文字）| 从文件里抠出来、能按字处理的纯文本 | `"退款期限是 7 天"` |
| `BytesIO` | **把内存里的一串字节伪装成"文件"** | 有些解析库只认文件、不认裸字节，BytesIO 就是"套个假文件袋再喂给它"，好处是不用先写硬盘 |

还有个撞名问题：Word 库和 LangChain 各有个 `Document` 类，文件开头 `from docx import Document as WordDocument` 给 Word 那个**改了名**区分。LangChain 的 `Document` 是全生态通用的"文字包裹箱"：正文装 `page_content`，标签装 `metadata`。

## 1. 门口贴的两张告示

```python
SUPPORTED_TYPES = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
```

只收这四种文件；单文件最多 10 MB（提取 PDF 很吃内存，防撑爆）。写成 `{...}` 集合是因为后面 `后缀 in SUPPORTED_TYPES` 判断又快又干净。

## 2. 两个后厨小工：`_read_pdf()` 和 `_read_docx()`

函数名前缀 `_` 是 Python 惯例："内部小工，外面别直接使唤"。正门只有一个 `extract_text()`。

### `_read_pdf(content)`：逐页抠字

```python
reader = PdfReader(BytesIO(content))
if reader.is_encrypted:
    raise ValueError("请先解密 PDF 再上传")
return "\n".join(page.extract_text() or "" for page in reader.pages)
```

- `BytesIO(content)`：字节装袋，递给 PdfReader；
- 加密 PDF 直接拒收（不猜密码），还给人话提示；
- 最后一条一行顶三条：逐页 `extract_text()` 抠字；**`or ""` 是点睛的小保险**——纯图片页（插图、扫描件）抠出来是 `None`，而 `None` 没法用 join 拼接会当场报错，`or ""` 把它变成"一行白页"继续走；最后所有页用换行拼成一整段。

**没有 OCR**：整本是照片的扫描 PDF 抠出来全是空，到 `extract_text` 会被"没有可提取的文字"拦下。OCR（图像识字）要你自己先做完再传。

### `_read_docx(content)`：按正文顺序读段落+表格

```python
for block in document.iter_inner_content():
    if isinstance(block, Paragraph):
        lines.append(block.text)
    else:
        for row in block.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
```

为什么用 `iter_inner_content()` 而不是只读 `paragraphs`？Word 正文实际是**段落和表格交替的序列**。只读段落，表格整个消失。`iter_inner_content` 按你肉眼看到的顺序一块块过。

表格行拼成 `张三 | 25 | 北京` 这样的文字——因为下游向量检索只吃纯文本，竖线让模型大致看出"这几个词同行、有关系"就够了。字体、颜色、图片全部丢弃：AI 只关心字，不关心排版。

## 3. 正门 `extract_text()`：安检 → 分诊 → 翻译错误 → 验货

上传线和启动线**都汇到它**。给文件名和字节，成功返回正文文字，不合格抛带人话的 `ValueError`。

### 第一步：查证件

```python
suffix = Path(name).suffix.lower()
if suffix not in SUPPORTED_TYPES:
    raise ValueError("仅支持 TXT、Markdown、PDF 和 DOCX 文件")
```

`.suffix` 取扩展名，`.lower()` 转小写——`笔记.PDF` 也能认出。注意这只是"按名字分诊"，**不能证明文件真没坏**（改名的图片照样能过这关，下一关才拆穿）。

### 第二步：按类型分诊

```python
if suffix in {".txt", ".md"}:
    text = content.decode("utf-8-sig")
```

TXT/MD 一行 decode 完事；`utf-8-sig` 里的 `-sig` 会自动吃掉 Windows 记事本爱在开头塞的隐形标记字节（BOM），防止正文第一个字变成怪符号。PDF/Word 请后厨小工。

### 第三步：三层错误翻译成人话（最值得学的部分）

```python
except UnicodeDecodeError:
    raise ValueError(f"{name} 必须保存为 UTF-8 编码") from None
except ValueError:
    raise
except Exception:
    raise ValueError(f"无法读取 {name}，请检查文件格式或文件是否损坏") from None
```

| 捕获到什么 | 意思是 | 处理 |
| --- | --- | --- |
| `UnicodeDecodeError` | TXT/MD 用了 GBK 等别的编码 | 换成可操作提示："存成 UTF-8" |
| `ValueError` | 自家代码**主动**抛的（"请先解密"已经写好了人话）| `raise` 原样放行，别重复包装 |
| 其他一切 | 文件损坏、伪装扩展名 | 统一提示"格式或损坏" |

两个语法点：

- `except ValueError: raise` 单独放行——若让它掉进最后的兜底被替换成"无法读取"，用户就丢了真正原因。顺序讲究：**专门错误先抓、自家错误放行、剩下全兜底**；
- `from None`——报错时不带原始异常链（那串"During handling of..."的内部调用栈）。对用户是噪音，还可能泄露内部库名。

### 第四步：验货

```python
if not text.strip():
    raise ValueError(f"{name} 没有可提取的文字；扫描 PDF 需要先做 OCR")
```

空文件、纯图片 PDF 在这拦下。为什么必须拦？空文字切出 0 个片段，知识库会出现"有这文档但永远搜不到"的幽灵状态。**在门口拦住，比在深处排查便宜一百倍**——这叫"尽早失败"。

## 4. 上传线 `save_uploads()`：整批验完，才准上架

参数 `files` 是 `[(文件名, 字节), ...]` 列表。灵魂设计一句话：**先验完整批货，再动架子一下**。

### 第一轮循环：只做检查，一个字都不写盘

| 检查 | 防什么 |
| --- | --- |
| 文件名不含 `\ / : < > " | ? *` | 这些是文件系统禁区；`/` 和 `\` 还能拼出路径 |
| 批内无重名（`name in names`）| 你一次选了俩 notes.txt，落盘时后者默默覆盖前者——宁可报错让你改名 |
| 每个 ≤ 10 MB | 防撑内存 |
| **`extract_text(name, content)` 试跑一遍** | 返回值直接扔掉！调它只为"提取失败就抛异常"。**写盘之前**先确认文件真能读——传个伪装成 .pdf 的图片，当场拒收，而不是"文件已存好，入库时才发现读不了"的半成品 |

先验批再写盘的效果：第二份文件不合格时，第一份也**不会**提前躺进仓库。

### 第二轮循环：真正落盘

```python
folder.mkdir(parents=True, exist_ok=True)
for name, content in files:
    target = (folder / name).resolve()
    if target.parent != folder.resolve():
        raise ValueError("上传文件必须保存在上传目录内")
    target.write_bytes(content)
```

- `Path / 字符串` 是**拼路径**（不是除法）；`.resolve()` 结算成绝对真实路径；
- `target.parent != folder` 专防**路径穿越**：文件名要是 `../../evil.txt`，拼接后 resolve 会把 `..` 真实结算出来，爹目录一查不是上传目录 → 拒收。前面已禁了 `/` `\`，这是双保险；
- **`write_bytes(content)`**：把内存里那串字节**原封不动**写成硬盘文件——存的是**原件**（PDF 还是完整 PDF），不是提取后的文字。以后重传更新、删除后再同步，操作的都是这份原件。对应地，读盘叫 `read_bytes()`，一对镜像动作。为什么不用 `write_text`？PDF/Word 是二进制，按"文本"写会解码再编码，中间坏掉；字节进字节出才无损；
- 同名覆盖是故意的：重传同名文件=更新原件，之后 `rag.py` 靠内容指纹自动发现切片变了，旧向量删、新向量进，不需要额外做版本管理。

**诚实边界**：这是"先检查再写盘"，不是磁盘事务。写到一半断电，已写入的文件不会自动撤销。

## 5. 启动线 `load_documents()`：全仓库盘点拆包

程序启动、点"同步知识库"、以及上传完之后（`ingest()` 内部），都走它。**每次把两个目录所有合格文件全部读一遍抠一遍**，返回统一的 Document 列表。

```python
folders = [(docs_dir, ""), (uploads_dir, "uploads/")]
for folder, prefix in folders:
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TYPES:
            continue
        text = extract_text(path.name, path.read_bytes())
        source = prefix + path.relative_to(folder).as_posix()
        documents.append(Document(page_content=text, metadata={"source": source}))
```

| 零件 | 白话 |
| --- | --- |
| 两个目录+前缀 | 预置资料来源叫 `course.md`，上传的叫 `uploads/notes.txt`——同一个向量库里，标签让人一眼分清出身 |
| `rglob("*")` | 递归扫描，子目录里的也收 |
| `sorted(...)` | **按名字排序**：切片编号（`course.md#0/#1/...`）取决于读取顺序，不排序的话每次扫描顺序漂移、编号对不上号 |
| 两个 `continue` | 目录本身、不认识的扩展名——**静默跳过** |
| `read_bytes()` | 硬盘原件 → 内存字节，和上传线汇成同一种原料 |
| `relative_to(folder).as_posix()` | 掐掉目录前缀只留相对路径，反斜杠统一成正斜杠（Windows/Linux 产出的来源名一致）|
| `Document(page_content, metadata)` | LangChain 标准包裹箱，装好文字+来源标签 |

静默跳过 vs 上传报错的不对称是有意的：上传是**用户当面操作**，失败了要有明确弹窗，宁可整批不动；盘点是**后台例行工作**，目录里混个 `.tmp` 不该阻塞启动。

一个诚实边界：启动线**没有 try/except**——预置目录里某个 PDF 真坏了，错误一路炸到启动流程，网页显示报错。自己的资料目录坏了就该知道并修，而不是假装没看见。

## 6. save_uploads vs load_documents：一个存、一个取（最常见困惑）

| | `save_uploads()` | `load_documents()` |
| --- | --- | --- |
| 方向 | **放进仓库**（写盘）| **从仓库搬出来**（读盘）|
| 输入 | 内存字节（刚选的网页文件）| 硬盘已有全部文件 |
| 输出 | 落盘的原件+名字列表 | Document 列表（纯文字包裹）|
| 何时跑 | 只有点"上传并入库"那一刻 | 启动/点同步/上传之后（由 ingest 调用）|
| 管几个 | 只这批新文件 | 全部文件重新盘点 |

上传一个文件时两者接力：

```text
点"上传并入库"
  → rag.upload() → save_uploads()：安检 + write_bytes 上架   【此刻只多了个文件，向量库没动】
  → rag.upload() → ingest() → load_documents()：全仓库盘点抠字
  → 切片算指纹 → 只有新切片花钱向量化 → 存 Chroma
```

为什么要"全量盘点"这么笨？**便宜**（就是读硬盘），而且把"同步"变成一条通用路径——上传、改文件后点同步、启动自动同步，全走同一条 `ingest → load_documents`，不用各写一套。真正省的是后面的向量化：指纹没变的切片不重新花钱。

## 7. 全文件一张表

| 成员 | 比喻 | 谁调用 |
| --- | --- | --- |
| `_read_pdf()` / `_read_docx()` | 后厨拆包工 | `extract_text` |
| `extract_text()` | **正门**：查证件→分诊→翻译错误→验空 | 两条线都用 |
| `save_uploads()` | 收货员：整批验完才上架 | `rag.upload()` |
| `load_documents()` | 盘点员：全仓扫描统一装袋 | `rag.ingest()` |

## 本篇要带走的三句话

1. **尽早失败**：坏文件在门口就拒掉，上传前甚至预演一遍提取；
2. **边界统一格式**：文件从网页来还是硬盘来、TXT 还是表格 Word，出门全是同一种 Document，下游只认一种包裹；
3. **错误信息是给人看的**：三层 except 翻译、`from None` 降噪，全为网页上那句"请先解密 PDF 再上传"，而不是天书堆栈。

继续读：[embeddings.py：给文字发"数字名片"](D:/study/AI/ai-agent-demo-study/docs/code-guide/embeddings.md)。
