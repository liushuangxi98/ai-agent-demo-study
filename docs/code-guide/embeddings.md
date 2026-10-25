# embeddings.py：给每段文字发一张"数字名片"

对应源码：[agent_demo/embeddings.py](D:/study/AI/ai-agent-demo-study/agent_demo/embeddings.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

你问"买课多久能退"，资料里写"购买后 7 天内可以申请退款"——**字面完全不同，意思却接近**。怎么让电脑找到这种资料？

办法是请一个专门模型给每段文字算一串数字（叫**向量**，就当是"数字名片"）：**意思越接近的文字，名片上的数字越接近**。这样"按意思搜索"就变成了"比数字"。

`embeddings.py` 干的事和 llm.py 一模一样——**只造电话机，不打电话**：创建一个调用百炼 Embedding 服务的客户端。不切文档、不存向量、不写答案。

## 1. 它和"问答模型"是两个工种，别混

| | 千问（ChatOpenAI，llm.py 造的）| Embedding（这里造的）|
| --- | --- | --- |
| 干的活 | 读懂问题资料，**写回答** | 把文字**变成一串数字** |
| 会不会聊天 | 会 | 完全不会，就是"文字→数字"的翻译机 |
| 输出 | 一段话 | `[0.12, -0.08, 0.43, ...]` |

同一个百炼地址、同一把密钥（`settings.llm_api_key` 两边共用），只是模型名不同（`text-embedding-v4`）。

向量里每个数字具体代表什么？没人说得清，也不用懂。**只需要记住一条：同一个模型算出来的向量，可以互相比远近。**

## 2. 它在项目里的位置（两条流水线都会经过）

```text
入库：原件 → documents.py 抠字 → rag.py 切片 → 【这里】算文档向量 → Chroma 存
提问：问题 → 【这里】算问题向量 → Chroma 比远近找片段 → 原文交给千问写答案
```

## 3. `build_embeddings()` 逐参数读

```python
return OpenAIEmbeddings(
    model=settings.embedding_model,        # 用哪个向量模型
    api_key=settings.llm_api_key,          # 和问答共用密钥
    base_url=settings.llm_base_url,        # 和问答共用地址（类名带 OpenAI 但打给百炼，同 llm.py 的兼容套路）
    request_timeout=30,
    max_retries=1,
    chunk_size=10,
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)
```

三个不显眼的参数：

| 参数 | 白话 |
| --- | --- |
| `chunk_size=10` | 一次请求**最多带 10 条文字**（批量发送省往返）。100 个切片分 10 批发 |
| `check_embedding_ctx_length=False` | 关掉客户端自带的"本地分词+超长自动切"流程。为什么关？文档已经被 rag.py 切成 400 字符的小段了，不需要再切一遍；而且百炼模型的分词器和客户端内置的那套不同，本地瞎检查反而添乱 |
| `encoding_format="float"` | 要求向量按普通浮点数返回（另一种 base64 压缩格式省流量但要解码）|

**注意**：关客户端检查 ≠ 文字无限长不炸。远端模型自己有输入上限，只是由远端报错而不是本地预防。

### 两个"chunk_size"撞车警告（最容易看混的坑）

| 位置 | 值 | 说的是哪件事 |
| --- | --- | --- |
| 本文件 `chunk_size=10` | 10 **条** | 每次向服务发几条文字（快递打包）|
| `rag.py` 的 `CHUNK_SIZE=400` | 400 **字符** | 每条文档片段切多长（裁纸条）|

先**把文章裁成 400 字的小纸条**（rag.py），再**每次拿最多 10 张纸条**去做名片（这里）。两个步骤、两种"块"，只是碰巧都叫 chunk。

## 4. 谁真正按下了拨号键？

客户端造好后其实有两扇门会触发网络请求：

```python
embed_documents(texts)   # 一批文字 → 一批向量（入库时用）
embed_query(query)       # 一个问题 → 一个向量（检索时用）
```

但你在 `rag.py` 里**找不到人调用它们**——因为 Chroma 被喂了这个客户端当"外挂"：

```text
store.add_documents(...)                → 内部自动调 embed_documents
store.similarity_search_with_score(...) → 内部自动调 embed_query
```

所以"没在代码里看到 embed_query"不代表没请求向量服务。读 LangChain 生态的代码要习惯这种**"把能力注入，框架替你调"**的模式。

## 5. 走一遍查退款

| 步 | 发生什么 |
| --- | --- |
| 1 | course.md 抠出正文，切成"购买后 7 天内…"等切片 |
| 2 | 新切片每 10 条一批请求向量，Chroma 存下（向量+原文+来源）|
| 3 | 你问"退款条件是什么"→ 问题请求一次向量（**每轮提问都花这一下**，闲聊和计算题也不例外，因为程序每轮强制先自动检索）|
| 4 | Chroma 比远近，返回最近的 3 段原文 |
| 5 | 千问读原文写答案，标 `[course.md#1]` |

问题向量用完即弃——它不是知识记录，不存 Chroma 也不进 MySQL。MySQL 存聊天是另一条流程。

## 6. 为什么换向量模型必须"换柜子"

不同模型出的名片**不能混着比**——就像两把刻度不同的尺子量出来的数没法直接对照。

`rag.py` 把"模型名+接口地址+切片参数"揉一起算出 12 位哈希当集合版本号（`learning_xxx`）。你哪天把 `EMBEDDING_MODEL` 换了，版本号变化 → 自动用一个全新空集合重新入库，绝不会出现"新问题向量 vs 旧模型文档向量"的关公战秦琼。旧集合不会自动删，可能还占着磁盘。

## 本篇要带走的三句话

1. Embedding = "文字→数字名片"的翻译机，按意思搜索全靠名片比远近；
2. 本文件只造电话；真正拨号的是 Chroma 内部代调，每轮提问都产生一次问题向量请求；
3. 模型换 = 尺子换 = 索引作废重建，这就是 rag.py 里版本哈希存在的原因。

继续读：[rag.py：图书馆管理员——切片、入库、检索](D:/study/AI/ai-agent-demo-study/docs/code-guide/rag.md)。
