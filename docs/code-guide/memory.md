# memory.py：档案室——流水账和档案卡的仓库管理员

对应源码：[agent_demo/memory.py](D:/study/AI/ai-agent-demo-study/agent_demo/memory.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

网页一关、程序一停，内存里的东西全丢。想下次还能翻旧聊天、还记得你更正过的名字，就得存进 MySQL 数据库。

`memory.py` 是**档案室管理员**，管两本账：

| 账本 | 表 | 内容 | 按什么找 |
| --- | --- | --- | --- |
| 流水账 | `conversation_turns` | 每轮的原问题、最终回答、工具、反思和用量记录 | 用户 + 会话 |
| 档案卡 | `user_memories` | 从聊天提炼的个人事实（姓名→小明）| 只按用户 |

**记住这个不对称**：流水账按会话隔离（新会话不带旧聊天），档案卡按用户共享（新会话照样知道你叫啥）。模型怎么"提炼事实"是 structured_memory.py 的事，这里只管"怎么存、怎么取"。

## 0. 六个前置概念（一张表过完）

| 名字 | 白话 |
| --- | --- |
| 表 / 行 / 列 | 表像 Excel，一行=一条记录，一列=一类内容 |
| ORM | "翻译官"：你用 Python 对象操作，它替你生成 SQL 发给数据库 |
| 模型类 | 描述某张表长什么样的模板类（`ConversationTurn` ↔ 聊天表）|
| Engine | 档案室的**总台**：存着连接配置、管理一池可复用的数据库连接 |
| Session | **一次办事窗口**：查询、登记要存的对象、跟踪修改，办完就关窗 |
| 事务 | 一组操作**绑成一单**：全成才提交，中途出错全部撤销（回滚）|

最容易认错的是 **Session**：这里的数据库 Session 和聊天会话（session_id）毫无关系，撞名而已。你聊一轮天，档案室会开**好几次**短暂的工作窗口，各用各的、用完即关。

```text
Engine 管连接 → session_factory（窗口开设机）→ 每次操作开一个 Session → 办事 → 关窗
```

## 1. `Base`：设计图收纳夹

```python
class Base(DeclarativeBase):
    """ORM 模型的公共基类。"""
```

本体啥也不干，作用只有一个：所有继承它的模型类，SQLAlchemy 都会把表设计图收进 `Base.metadata`（"所有表的设计图合集"）。之后 `Base.metadata.create_all(engine)` 一喊，照着设计图把缺的表建出来。定义类 = 画图，**没有连数据库**。

## 2. `ConversationTurn`：流水账里"一轮"长什么样

```python
class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
```

`__tablename__` = 对应的真实表名；`Mapped[int]` = Python 侧这属性是整数；`mapped_column(...)` = 这列在数据库里的规矩。

### 九列逐一看

| 列 | 白话 | 备注 |
| --- | --- | --- |
| `id` | 本轮编号 | 主键+自增：新行编号数据库自己排，绝对不重 |
| `user_id` | 谁的记录 | 特殊：单独用 `utf8mb4_bin` 排序规则 = **大小写敏感**（`Tom`≠`tom`）|
| `session_id` | 哪个会话 | |
| `user_text` | 原问题 | 存用户原话，不混检索资料 |
| `assistant_text` | 最终回答 | |
| `trace_json` | 工具和反思 | LONGTEXT，存的是 JSON **文字** |
| `usage_json` | 本轮 `current` 与会话累计 `total` | 原生 JSON，Python 直接用字典；没记录时为数据库 NULL |
| `summary_json` | 累计摘要与覆盖位置 | 可空原生 JSON，未摘要时为 SQL NULL |
| `created_at` | 保存时间 | 由 MySQL 生成 |

细节：`Mapped[...]` 没写可选的列默认**不允许 NULL**；`usage_json` 与 `summary_json` 明确可空。空字符串 `""` 是合法值，NULL 不是——两回事。

`__table_args__` 里那条：

```python
Index("ix_user_session_id", "user_id", "session_id", "id")
```

**索引=书柜侧面的分类标签**。按"用户+会话+编号"查时，MySQL 不用一页页翻全表，顺着标签直达。纯性能优化，不改变任何数据。（别和向量检索的"索引"混——那是按意思找，这是按字段精确捞。）

### `to_dict()`：出仓前打包

```python
return {"id": self.id, "user_id": self.user_id, ...}
```

查到的 ORM 对象挂在 Session 窗口上，**窗口一关有的属性就取不到了**。所以趁窗口开着，把包括用量和摘要在内的 9 个字段抠进普通字典带走。外面网页和 Agent 只见字典，完全不用懂 ORM 和连接——脏活不出档案室。

## 3. `UserMemory`：一人一卡的档案卡

| 列 | 白话 |
| --- | --- |
| `user_id` | 主键 → **一人最多一行** |
| `facts_json` | 整份画像一个 JSON 列 |
| `updated_at` | 最后一次写卡时间 |

没有 session_id——所以画像天生跨会话共享。

JSON 列的小幸运：Python 里直接给**字典**，SQLAlchemy 自动和数据库 JSON 互转，不用自己 `json.dumps`。对比聊天表 `trace_json` 是文字列，就得手工 dumps——两条列类型两种待遇。

## 4. `validate_session_id()`：门口保安

```python
if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
    raise ValueError("会话 ID 只能包含 1～64 个字母、数字、下划线或连字符")
return session_id
```

和 config.py 查 USER_ID 同一招：先卡格式，后面的方法就能少写重复检查。现在 `chat()` / `resume()` 在入口校验一次，`recent()` 直接服务网页也保留；`save_turn()` / `context()` 只由已校验的入口调用，不再重复校验。它只查**格式**，不查"这会话在库里存不存在"（网页加载旧会话时会另外查有没有聊天）。

## 5. `ConversationMemory.__init__()`：开档案室（五步，步步有讲究）

### 第一步：造连接地址（故意不带库名）

```python
server_url = URL.create("mysql+pymysql", username=..., password=...,
                        host=..., port=..., query={"charset": "utf8mb4"})
```

`mysql+pymysql`=连 MySQL、走 PyMySQL 驱动。用 `URL.create` 而不是手拼字符串：**密码里带 `@`、`:` 这类特殊字符时，手拼会把地址拆错位**。

地址故意不写库名——因为目标库可能还不存在，得先摸到 MySQL 服务门口再说。

### 第二步：两个连接保险

```python
options = {"pool_pre_ping": True, "connect_args": {"connect_timeout": 5}}
```

`pool_pre_ping`：从连接池领连接前先 ping 一下——MySQL 会掐断闲置太久的连接，不检查就会拿到根"死电线"。`connect_timeout=5`：建线最多等 5 秒（只管连不上，不是所有 SQL 的总时限）。

`**options` 是"把字典拆成具名参数传进去"，不是乘法。

### 第三步：先连服务，把库建了

```python
server = create_engine(server_url, **options)
with Session(server) as session:
    session.execute(CreateSchema(database, if_not_exists=True))
    session.commit()
finally:
    server.dispose()
```

MySQL 术语里 Schema≈数据库。`if_not_exists=True`：有就跳过，没有才建。这段临时连接用完立即 `dispose()` 扔掉。

建库是 DDL（定义结构），没有"对象"可以 `session.add`，所以这里直接 `execute` 一句建库指令——ORM 也留了这扇直门。

### 第四步：换上目标库地址，建表

```python
self.engine = create_engine(server_url.set(database=database), **options)
Base.metadata.create_all(self.engine)
```

`.set(database=...)` 返回加了库名的**新地址**（不是 SQL 的 SET）。`create_all()` 照设计图建**缺的**表和索引——**已有的表不清空、不改结构**。所以改代码字段后重启不会丢聊天，但老表也不会自动长出新列；本次选择从空表重新开始，不在每次启动时搬旧数据；已有旧表需要一次性重建。

建表失败 → 释放刚建的连接池 → 继续抛错。别留下"连接开一半"的烂摊子。

### 第五步：备好"开窗口"的机器

```python
self.session_factory = sessionmaker(bind=self.engine)
```

工厂本身不产生窗口，只记住"以后开窗口都绑这台 Engine"。以后每调一次 `self.session_factory()` 才开一个新窗口——**每个操作各用各的窗口**，两个网页请求绝不共用一个 Session（那会把彼此未提交的修改搅在一起）。

## 6. `profile()`：取档案卡（最短的读，信息量不小）

```python
with self.session_factory() as session:
    profile = session.get(UserMemory, self.settings.user_id)
    return profile.facts_json if profile is not None else {}
```

`session.get(类, 主键)`=按主键点行，没有返回 `None`。`A if 条件 else B`=有卡抄卡内容，没卡给**空字典**（给 None 的话调用方还得防，统一给 `{}` 最省心——空画像就是"啥都不知道"，语义天然正确）。

只看 user_id，不看 session_id → 画像跨会话。`with` 结束自动关窗。

## 7. `save_turn()`：一单双货——流水和档案卡同存同滚

参数：会话号、原问题、回答、轨迹，可选的画像变化 `changes`、时间 `now`、用量字典 `usage`（`current` 是本轮，`total` 是当前会话累计）。`usage` 要按名字传，单独写入 `usage_json`。

### 第一二步：验号、造对象（注意：还没写库！）

```python
now = now or datetime.now().astimezone()
turn = ConversationTurn(user_id=..., ..., trace_json=json.dumps(trace, ensure_ascii=False), usage_json=usage)
```

`now or 现取`：调用方给了就用给的，没给才现取。`ensure_ascii=False`：中文别转义成 `\uXXXX`，肉眼可读。此刻 `turn` 只是内存对象，数据库毫不知情。

### 第三步：开一个"绑成一单"的事务

```python
with self.session_factory.begin() as session:
    session.add(turn)
    if changes:
        session.flush()
        self._update_profile(session, changes, turn.id, now)
```

`begin()` 版 with：**正常走出 = commit 提交；半路抛错 = rollback 全撤**。

| 动作 | 白话 |
| --- | --- |
| `session.add(turn)` | 把新流水**登记**给 ORM"待写入"，还没落库 |
| `if changes:` | 有画像变化才走下面（空列表直接跳过，不建空卡）|
| `session.flush()` | **提前执行**待写入的 INSERT——为的是让数据库把自增 `id` 回填到 `turn.id`。注意：**flush ≠ commit**，整单还没生效 |
| `_update_profile(...)` | 拿着刚到账的真实 `turn.id`，在**同一单里**改档案卡 |

为什么非要 flush？画像每条事实要记"来源对话 ID"（`source_turn_id`），得先知道本轮聊天 ID 是几号。像先打出一张收据拿单号，再凭单号登记附属业务——**但整单未结账，后面一步崩，刚写的收据也一起撕掉**。这就是"问答和画像一起成功或一起失败"的实现。

## 8. `_update_profile()`：改卡四步（并发防御的精华段）

由 save_turn 内部调用，**复用同一个窗口、同一单事务**，不自己提交。

### 第一步+第二步：没卡先建卡，撞车也不怕

```python
if session.get(UserMemory, user_id) is None:
    try:
        with session.begin_nested():
            session.add(UserMemory(user_id=user_id, facts_json={}))
            session.flush()
    except IntegrityError as error:
        if error.orig.args[0] != 1062:
            raise
```

场景：同一用户的两个请求同时进来，**都读到"还没有卡"**，都去建卡。后插的那个会撞主键（MySQL 错误码 1062=重复键）。

`begin_nested()` = 在大单里设一个**局部撤销点（保存点）**：撞车时只撤销"建卡"这一步，**大单里的聊天 INSERT 毫发无损**。撤销后不慌——别人已经把卡建好了，我往下直接拿卡就行。错误码不是 1062 的完整性问题（比如字段非法）照样上抛，不装死。

### 第三步：先锁卡、再拿最新版

```python
profile = session.get(UserMemory, user_id, with_for_update=True, populate_existing=True)
```

| 参数 | 白话 |
| --- | --- |
| `with_for_update=True` | 给这行上**写锁**：我改卡期间，别的请求想改同一张卡得排队等我这单结束 |
| `populate_existing=True` | 就算这窗口之前见过这张卡，也强制从库重读最新版，别拿缓存的旧卡改 |

这两个参数合起来防的是**覆盖事故**：A 请求要改"姓名"，B 请求要改"学习目标"。不锁不重读的话，俩人各拿旧卡改自己那格再整卡盖回去——后提交的把前一个的修改抹掉了。锁住+拿最新，B 改的卡里就有 A 刚写的姓名。**"合并前先拿最新"** 是并发编程的第一军规，这段就是活教材。

找不到卡（理论上不可能）→ `RuntimeError`，绝不假装成功。

### 第四步+第五步：整卡换新，回大单结账

```python
profile.facts_json = merge_profile(profile.facts_json, changes, turn_id, now)
profile.updated_at = func.current_timestamp()
```

两个细节：

1. **赋一个全新的字典**，而不是往旧字典里改深层键。SQLAlchemy 跟踪"属性被赋值"很在行，跟踪"字典内部第 3 层某个键变了"很不在行——整卡替换它 100% 能察觉。这是操作 JSON 列的通用保命写法；
2. `func.current_timestamp()` = "更新时间用 **MySQL 服务器的钟**"，不用 Python 进程的钟——多机器部署时少一个时钟不一致的坑。（对照：事实自己的 `recorded_at` 用的是 Python 本地时间，两套时间来源，README 也特意提了。）

之后保存点结束不等于全单生效，最终由 save_turn 外层 with 一起结账。若 `merge_profile` 抛"超过 20 格"，本单全回滚，chat() 那边接着只存问答——流水和档案卡的"连坐"到此为止。

## 9. `recent()`：取最近 N 轮（一条查询单的诞生与执行）

```python
statement = (select(ConversationTurn)
    .where(ConversationTurn.user_id == self.settings.user_id,
           ConversationTurn.session_id == session_id)
    .order_by(ConversationTurn.id.desc())
    .limit(limit))
with self.session_factory() as session:
    turns = session.scalars(statement).all()
    return [turn.to_dict() for turn in reversed(turns)]
```

| 零件 | 白话 |
| --- | --- |
| `select(类)` | "我要完整的聊天对象"（查询单只写好，还没执行！）|
| `.where(a, b)` | 逗号=AND：当前用户 且 当前会话 |
| `.order_by(id.desc())` | 新号在前——配合 limit 才能"截最近 N 条" |
| `.limit(limit)` | 最多 N 条 |
| `session.scalars(...).all()` | **这行才真发 SQL**，收回对象列表 |
| `reversed(...)` | 拿回是 [5,4,3]（新→旧），翻成 [3,4,5]（旧→新）——聊天得按时间正序 |
| `to_dict()` | 趁窗口没关打包带走 |

"先倒序取最新，再翻正"是取"最近 N 条且按时间正序"的标准舞步。

## 10. `list_sessions()`：会话目录（分组统计）

侧栏"已保存会话"下拉框的数据源。没有会话表，从流水账现算：

```text
当前用户的记录 → 按 session_id 分组 → 数每组轮数 count() / 找最新时间 max(created_at)
→ 按 max(id) 倒序（最近活跃的会话排前）→ limit
```

统计行不对应"某一条完整聊天"，所以用 `session.execute(...).all()` 收多列结果，手动组装字典——和上面 `scalars` 收对象是两种读法。`label("turn_count")` 给统计列起名，`row.turn_count` 取。

顺带：刚点"新建会话"没聊过 = 流水账零行 = 不会出现在列表里，天然正确。

## 11. `context()`：读取摘要与其后全部原文

```python
turns, previous_usage, summary_state = self.memory.context(session_id)  # 三项数据各自承担职责。
```

| 返回值 | 内容 | 交给谁 |
| --- | --- | --- |
| `turns` | 未进入摘要的完整问答字典，保留真实 `id` | Agent 组装历史消息、确定下一次摘要边界 |
| `previous_usage` | 最后一轮 `usage_json.total` | 现有用量统计 |
| `summary_state` | `{text, through_turn_id}`；未摘要为 `{}` | Agent 组装摘要资料消息与继续压缩 |

读取顺序：先按 `user_id + session_id` 找最后一行，再取它的累计和摘要。会话 ID 由 Agent 入口校验，这里直接查询；没有记录返回 `([], {}, {})`，没有摘要从 ID 0 开始，即读取本会话全部原文。

有摘要时，查询 `id > through_turn_id` 且 `id <= latest.id` 的全部记录，按 ID 正序返回。上下界与本次快照保持一致，页面 `recent()` 的 50 条限制不参与这里的读取。数据库 ID 是全表编号，跨会话可能不连续，不能用轮数代替。

摘要正文和游标合并成一次有效性检查；损坏快照会明确报错，避免悄悄跳过历史。返回普通字典后，Session 随 `with` 结束关闭，模型请求不会占着数据库事务。

### 摘要如何随问答保存

`ConversationTurn.summary_json` 是可空原生 JSON。`save_turn(..., summary=state)` 把供下一轮使用的快照与问答、轨迹、用量、画像变化同事务保存。没有重新摘要时传入原快照；没有摘要时传 `None`。

`save_turn()` 本身不调用模型。直接写入测试问答不能证明摘要生成已经验证。摘要失败时由 `chat()` 继续传原快照，容量回滚后的备用保存也要传同一份摘要和用量。

### 缺列或表结构不兼容时如何处理

`create_all()` 只建缺失表，不修改旧表。用户已明确要求：缺列等不兼容结构问题，由开发阶段清空本项目业务表和生成缓存，再按当前 ORM 模型重建；不保留手工加列要求或迁移兼容逻辑。普通启动只建表，不重复清空数据。本次重置后两表为空，摘要列已存在。

## 12. `close()`：关机收线

`engine.dispose()` 关掉连接池的空闲连接。**不删库、不删聊天**。初始化后续步骤失败时的清理、评测收尾都靠它。

## 13. 走一遍"更正姓名"的完整落库

| 步 | 发生什么 |
| --- | --- |
| 1 | agent 提取出变化单：姓名 → 小舟（证据"更正：我叫小舟"）|
| 2 | save_turn 造本轮聊天对象，`begin()` 开事务 |
| 3 | `add` 登记；`flush()` 让 INSERT 先执行，本轮拿到真实 id=56 |
| 4 | _update_profile：上写锁 + 重读最新卡 |
| 5 | merge_profile 复印卡、只盖姓名格，`source_turn_id=56` |
| 6 | 整卡赋回 facts_json，updated_at 用 MySQL 时钟 |
| 7 | 走出 with → **一次 commit，流水+档案卡同时生效** |
| 8 | 新旧会话下轮各自 `profile()`，都读到小舟 |

若第 5 步画像超 20 格抛错：第 3、6 步全部回滚，像什么都没发生；agent 再发起第二次只存流水。

## 本篇要带走的三句话

1. 两张表两种归属：流水账=用户×会话隔离，档案卡=用户共享——"换会话还记得名字"的全部秘密；
2. 一个事务绑两货：add→flush 拿号 →锁卡→合并→commit，问答和画像要么同活要么同死；
3. 并发三连（保存点防建卡撞车、行锁防覆盖、重读拿最新）+ 整卡赋值 + 打包出仓（to_dict），是这个文件里最值得抄进笔记的工程手法。

继续读：[structured_memory.py：档案卡书记员](D:/study/AI/ai-agent-demo-study/docs/code-guide/structured_memory.md)，再看 [agent.py：把一轮问答串起来](D:/study/AI/ai-agent-demo-study/docs/code-guide/agent.md)。
