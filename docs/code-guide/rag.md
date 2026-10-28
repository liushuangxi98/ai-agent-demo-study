# rag.py：图书馆管理员——裁纸条、建索引、按意思借书

对应源码：[agent_demo/rag.py](D:/study/AI/ai-agent-demo-study/agent_demo/rag.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

documents.py 把文件抠成了整段文字，但一本书几十万字，不能整本塞给模型。`rag.py` 是**图书馆管理员**，管两件事：

- **存书**（ingest）：把长文裁成小纸条、给纸条做数字名片、放进 Chroma 这个书架；
- **借书**（retrieve）：你的问题做成名片，去书架找最像的几张纸条，带来源交出去。

它**不负责写答案**——找到的资料由 `agent.py` 交给千问。这条"先找资料再回答"的路子就叫 RAG（检索增强生成）。

| 方法 | 白话 |
| --- | --- |
| `__init__()` | 开馆：备好书架、挂上"名片机" |
| `_prepare_chunks()` | 裁纸条、贴标签、算指纹（不花钱）|
| `ingest()` | 对账：新旧清单一比，只给新纸条做名片 |
| `upload()` | 收新书：存原件，再走一遍对账 |
| `delete()` | 删一份知识文档：删原件，再对账清切片 |
| `retrieve()` | 按问题借纸条 |
| `context()` | 借书+排版一条龙，给工具直接用 |

## 0. 先分清"入库"和"提问"这两条完全不同的线

```text
入库（启动时/上传时/点同步时）：
  原件 → 抠字 → 裁400字纸条 → 算指纹 → 只对"没见过的"纸条花钱做名片 → 上架

提问（每轮）：
  问题 → 花钱做一张问题名片 → Chroma 比远近 → 带回最多3张相近纸条 → 太远的扔掉
```

重启会重新走一遍入库**流程**，但没变的纸条不会重新花名片钱。**每轮提问的名片费则次次都花**（哪怕你问的是"现在几点"，因为程序强制每轮先自动检索一次）。

## 1. `__init__()`：开馆三件事

```python
self.uploads_dir = settings.data_dir / "uploads"
self.lock = threading.RLock()
```

第一件：记住原件在哪。第二件：**发一把锁**。改索引、查书架时先拿锁，防止两个线程同时翻架子把书弄乱。用 `RLock`（可重入锁）而不是普通锁的原因：`upload()` 自己拿了锁，里面还要调 `ingest()`（它也要拿锁）——同一人重复进同一房间不该把自己锁死，RLock 允许。

第二件是给配置算"版本标签"：

```python
identity = f"{settings.embedding_model}:{settings.llm_base_url}:split-{CHUNK_SIZE}-{CHUNK_OVERLAP}"
version = hashlib.sha256(identity.encode()).hexdigest()[:12]
```

把"用哪个向量模型+哪个接口+怎么切纸条"揉成一句话，算 SHA-256 哈希取前 12 位。**哈希=指纹**：输入哪怕改一个字符，指纹面目全非；输入相同，指纹永远相同。

第三件：开 Chroma 书架，集合名 `learning_<指纹>`。任何一个配置变了（换模型=换尺子，改切片=换裁法），书架自动换成全新空柜，逼着全量重建——宁可重做，绝不拿新旧名片混着比。旧柜子不自动删（占磁盘，可手动清）。

`persist_directory=data_dir/chroma` 表示书架记在硬盘上，重启书还在；`hnsw:space: cosine` 表示比名片用"夹角"（距离 0=意思一样，2=截然相反）；`anonymized_telemetry=False` 关掉 Chroma 的匿名使用统计。

### Windows 还要先准备一个临时目录

创建 Chroma 前，代码在 Windows 下做这几步：

```python
temp_dir = settings.data_dir / "tmp"
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(temp_dir.resolve())
os.environ["TMP"] = str(temp_dir.resolve())
```

`os.name == "nt"` 表示 Windows。`resolve()` 得到完整路径。这里设置的是当前 Python 进程使用的临时目录，后续 Chroma 创建 SQLite 临时文件时会用到。

本机曾出现：索引目录可写、SQLite 文件完整，但默认临时目录下 Chroma 仍报 `unable to open database file`。改用项目内目录后初始化和本地检索成功，因此先备好它再打开书架。

`data/tmp` 放临时文件，`data/chroma` 放长期索引，各管各的。首次初始化失败后要停止旧进程再运行，单纯刷新可能复用失败的内部对象。故障堆栈保存在 [本地日志](D:/study/AI/ai-agent-demo-study/docs/code-guide/logging_config.md)。

## 2. `_prepare_chunks()`：裁纸条（这步一分钱不花）

```python
documents = load_documents(self.settings.docs_dir, self.uploads_dir)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=60,
    separators=["\n\n", "\n", "。", " ", ""],
)
```

- 400：每张纸条最多约 400 **字符**（不是 token，中文一个字≈一个字符）；
- overlap 60：**相邻纸条头尾重叠约 60 字**。想象切香肠时两刀之间故意留点叠肉——万一"7 天内且学习进度不超 20%"这句话恰好被切断，它的内容会同时活在前后的纸条里，不至于两头都查不到；
- separators：切的时候**优先在体面的位置下刀**——先试空行，不够再试换行、句号、空格，全都不管用才硬切。"Recursive"就是按这个顺序表一层层试，"不够细就再细分"。

```python
for document in documents:
    for number, chunk in enumerate(splitter.split_documents([document])):
        chunk.metadata["chunk"] = number
        identity = f"{chunk.metadata['source']}:{number}:{chunk.page_content}"
        ids.append(hashlib.sha256(identity.encode()).hexdigest())
```

两个巧思：

1. **每份文件单独从 0 编号**（外层循环里每换文件就重新 enumerate）。你新增一个文件，不会导致别的文件切片编号全漂移；
2. **指纹 = 来源+编号+正文** 三样合算。三样都没变，指纹不变；改正文、改文件名、挪位置（编号变）→ 新指纹。"有没有变"就这样变成了简单的"指纹在不在旧清单里"。

诚实提醒：改了文章**开头**加一段话，后面所有纸条的内容和编号都可能连锁变化——重新做名片的不只你改那句。这是裁纸的固有代价，不是 bug。

最后：两张目录都没纸可裁时返回两个空列表。空库是允许的状态，`ingest()` 会清掉旧切片；这样删除最后一份文档后，应用仍能正常启动和问答。

## 3. `ingest()`：拿新旧清单对账

```python
with self.lock:
    chunks, ids = self._prepare_chunks()
    old_ids = set(self.store.get()["ids"])
    if not chunks:
        if old_ids:
            self.store.delete(ids=list(old_ids))
        return 0
    new_chunks, new_ids = [], []
    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in old_ids:
            new_chunks.append(chunk); new_ids.append(chunk_id)
    if new_chunks:
        self.store.add_documents(new_chunks, ids=new_ids)
    obsolete_ids = old_ids - set(ids)
    if obsolete_ids:
        self.store.delete(ids=list(obsolete_ids))
```

先拿锁，再读原件和切片，直到索引更新完成才放开。这是为了避免“旧同步先读旧文件，后来又把新上传的索引覆盖回去”。锁只协调同一进程中使用这个知识库对象的线程，不是跨进程的文件锁。

**没有文档时**，把旧索引全部清掉并返回 0：删除最后一份文档、目录暂时为空，都靠这一步收尾。

对着代码读这段对账账本：

| 情况 | 处理 | 花钱吗 |
| --- | --- | --- |
| 新旧清单都有 A、B | 原样留着 | 不花 |
| 只有新清单有 D（你新传的/改过的纸条）| `add_documents`：做名片+上架 | 只花 D 的 |
| 只有旧清单有 C（你删了文件/改了内容后的旧纸条）| `delete` 下架 | — |
| 两份清单都空 | 清空旧切片，返回 0 | — |

- `zip(chunks, ids)` 拉拉链：两列表按位置配对，第 i 张纸条配第 i 个指纹；
- `old_ids - set(ids)` 集合差："旧有、新没有"=过时；
- **先添加、后删除**：万一新纸条做名片失败（断网），旧知识还躺在架上；反过来先删后加，失败时你就是裸的；
- 返回的 `len(chunks)` 是**同步后的总纸条数**，不是本次新增数——网页上"共 N 个切片"说的是总量。

边界要说清：添加和删除是两步操作，中间没有一个"全成或全回滚"的总事务。极端情况下（加成功、删失败）可能旧纸条多留一阵。重启再同步一次，对账会把它找补回来。

## 4. `upload()`：收新书 = 存原件 + 走老流程

```python
with self.lock:
    names = save_uploads(files, self.uploads_dir)
    count = self.ingest()
return {"files": names, "chunks": count}
```

没有为上传另写一套入库——先让 documents.py 把原件存好，然后**复用刚讲的对账流程**。内外都要拿锁，所以才用可重入的 RLock。

断点续传式的安心设计：原件先存上了，就算后面做名片时断网，文件不会丢；修好网络点"同步知识库"再走一遍 ingest 就行。

## 5. `delete()`：删一份知识文档

`delete_knowledge` 工具最终调用的就是它：

```python
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
target.unlink()
count = self.ingest()
```

- `course.md`、`agent.md` 指预置资料；`uploads/notes.txt` 指上传文件；
- 绝对路径、`..`、目录外路径一律拒绝，防止删到 `.env` 或源码；
- 删的是**原件**，再靠 `ingest()` 的指纹对账移出旧切片；下次同步不会又加回来；
- 删掉预置资料后，可以用 git 恢复。

## 6. `retrieve()`：按问题借书

```python
if self.is_empty():
    return []
matches = self.store.similarity_search_with_score(query, k=self.settings.rag_top_k)
```

- 空架不再当成事故：返回空列表，`context()` 会回一句“知识库为空”，整轮问答继续；
- 有架但本题没查到是常态；
- 把问题递给 Chroma：内部自动做问题名片（这就是 embeddings.md 说的"框架代调"）、比远近、带回最近的 k 个候选（默认 3）。

```python
for document, distance in matches:
    if distance <= self.settings.rag_max_distance:
        hits.append({...})
```

**过一道"够不够像"的门槛**（默认 0.8，cosine 距离越小越像）。这一步很重要：哪怕书架上只有不相干的纸条，向量搜索也一定"最近的就是它们"——不过滤的话，AI 会拿到驴唇不对马嘴的资料硬编。先找 3 个、再过滤，所以最后可能剩 1 个，也可能一个不剩。

距离 0.8 是"资料像不像"的闸门，**不是"答案对不对"的正确率**。

每条命中带四个标签交给上层：`source` 哪份原件、`chunk` 第几号纸条、`distance` 多远、`text` 原文。

## 7. `context()`：检索与排版一条龙

```python
if self.is_empty():
    return "知识库为空，请先上传或同步文档。"
hits = self.retrieve(query)
if not hits:
    return "知识库未找到足够相关的证据。"

passages = [f"[{hit['source']}#{hit['chunk']}]\n{hit['text']}" for hit in hits]
return "\n\n".join(passages)
```

排版结果就是你答案里见过的样子：

```text
[uploads/notes.txt#0]
极光计划口令是蓝色松果。
```

这几行拼接直接放在 `context()` 中，不再额外跳到另一个排版方法。列表推导式逐段生成 `passages`，`"\n\n".join(...)` 用空行把各段连接起来。来源编号的样子没有改变。

`context()` 只是把两步连起来：空库 → 回“知识库为空”；查到 → 排版；没查到 → 回一句"知识库未找到足够相关的证据。"**这句提示本身就是给模型的证据**："书架没有依据"这个事实也 informs 模型别硬编人物/课程类事实。检索查到与否不禁止模型答常识、不禁止它用计算器；但接口真炸了会一路抛异常，本轮问答中止，不会带病继续。

## 8. 走一遍：从上传到答案的完整接力

| 步 | 谁 | 干什么 |
| --- | --- | --- |
| 1 | 你 | 上传写着"口令：蓝色松果"的 notes.txt |
| 2 | rag.upload | 存原件 → ingest 对账 |
| 3 | _prepare_chunks | 裁成 1 张纸条，算出指纹 |
| 4 | Chroma+embeddings | 新指纹 → 花钱做名片 → 上架 |
| 5 | 你 | 问"极光计划口令是什么？" |
| 6 | agent.chat | 直接调 knowledge.context("...") |
| 7 | retrieve | 问题做名片 → 找回 notes.txt#0（距离 0.2）|
| 8 | context 内部拼接 | 排出 `[uploads/notes.txt#0] 极光计划口令是蓝色松果。` |
| 9 | 千问 | 读资料写答案，标 `[uploads/notes.txt#0]` |

知识库是**全项目共用一个书架**：不分 USER_ID，换用户不换柜子；你的聊天画像归 MySQL 另一套管。

## 本篇要带走的三句话

1. 指纹对账是本文件的心脏：内容变没变 = 指纹在不在旧清单，省钱和增量全靠它；
2. 先加后删、空库清索引、版本哈希换柜——三处都是"宁可多做防御，不给事故开门"；
3. 检索永远会还你"最像的 k 个"，**距离阈值才是说"不"的那个人**。

继续读：[memory.py：档案室——聊天记录和画像的仓库](D:/study/AI/ai-agent-demo-study/docs/code-guide/memory.md)。
