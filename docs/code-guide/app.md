# app.py：网页怎么画，按钮点下去又做了什么

对应源码：[app.py](D:/study/AI/ai-agent-demo-study/app.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 1. 先知道这个文件负责什么

`app.py` 是用户操作项目的入口：画页面、接收点击和问题、调用后台，再显示结果。真正的检索、模型与工具循环、画像提取和保存，在 `agent_demo/` 里的模块完成。

建议先打开文件末尾的 `main()`。它把一整页按顺序串起来，看懂这里后，再顺着函数名回头看细节。

源码注释只保留函数用途和关键行为，详细的语法解释放在这份文档里，避免读主流程时被大段注释打断。

| 文件里的区域 | 函数 | 一句话用途 |
| --- | --- | --- |
| 公共函数 | `get_app()`、`show_error()` | 取得缓存的 Agent，统一显示错误 |
| 操作处理 | `new_session()`、`switch_session()` | 修改会话状态，准备提示 |
| 操作处理 | `handle_upload()`、`handle_sync()`、`handle_question()` | 执行上传、同步或一轮问答 |
| 页面绘制 | `render_usage()`、`render_profile()`、`render_sidebar()`、`render_chat()` | 显示用量和画像、画控件，返回本轮用户的操作 |
| 主流程 | `main()` | 先准备应用，再画页面、处理操作、显示提示 |

现在侧边栏的默认样子是：

```text
会话
  切换会话下拉框
  新建会话按钮
  简短的自动保存说明
  ▸ 知识库管理（默认收起）
  运行信息（直接显示用户、模型和数据库）
  会话提示
```

侧栏去掉了顶部的大块留白，会话标题从靠近顶部的位置开始显示。要上传或同步，展开“知识库管理”；运行信息直接显示，无需点击。展开后内容较多，或者浏览器窗口很矮时，侧边栏仍然可以滚动。

## 2. Streamlit 的“重跑”是什么意思

启动命令仍然是 `streamlit run app.py`。Streamlit 启动网页服务，用户提交问题、点击按钮或切换选项后，会重新执行页面脚本。

每次重跑都会重新定义前面的函数，走到文件最后，再调用 `main()`。**定义函数只是准备好步骤，调用函数才真正执行里面的代码。**

```python
if __name__ == "__main__":
    main()
```

这两行的意思是：这个文件作为页面脚本执行时，进入 `main()`；其他代码 `import app` 时，只加载定义，不主动画页面或创建 Agent。加了入口判断，仍然要用 Streamlit 启动，普通 `python app.py` 不会创建网页服务。

重跑带来两个问题，代码分别处理：

| 问题 | 怎么处理 |
| --- | --- |
| 普通变量会重新赋值，怎样记住当前会话？ | `st.session_state` 保存当前浏览器会话的状态 |
| 每次都创建 Agent，会不会反复连库和同步知识？ | `@st.cache_resource` 缓存 `get_app()` 的结果 |

`session_state` 不是数据库。它能跨脚本重跑保留状态，但不保证关闭页面或刷新浏览器后恢复。历史聊天能重新找回来，靠的是 MySQL 和会话下拉框。

## 3. 先读 main()：一整页怎么串起来

下面就是当前的主流程：

```python
def main() -> None:
    """每次页面重跑：准备资源 → 侧栏 → 聊天 → 提示。"""
    setup_logging()  # 先开日志，初始化失败也能留下完整堆栈。
    st.set_page_config(page_title="AI Agent 学习实验室", page_icon="🤖", layout="wide")
    # 侧栏内容从顶部开始；折叠按钮留在右上角，不再单独占一行。
    st.html("""
        <style>
        [data-testid="stSidebarHeader"] {
            position: absolute;
            z-index: 1;
            top: 0;
            right: 1.25rem;
        }
        [data-testid="stSidebarUserContent"] {
            padding-top: 0.5rem;
        }
        [data-testid="stSidebarUserContent"] h2 { padding-top: 0; }
        </style>
    """)
    st.title("🤖 AI Agent 学习实验室")

    # 1. 取得缓存的 Agent，启动失败就停止本次页面执行。
    try:
        app = get_app()
    except Exception as error:
        show_error(error)
        st.stop()

    # 2. 首次打开时生成会话号，后续重跑继续使用它。
    if "sid" not in st.session_state:
        st.session_state.sid = uuid4().hex

    # 3. 绘制侧栏，再处理上传或同步；操作结果也显示在侧栏。
    with st.sidebar:
        files, upload_clicked, sync_clicked = render_sidebar(app)
        if upload_clicked:
            handle_upload(app, files)
        if sync_clicked:
            handle_sync(app)

    # 4. 先显示历史问答，再处理新问题。
    try:
        question = render_chat(app)
        if question:
            handle_question(app, question)
    except Exception as error:
        show_error(error)

    # 5. 最后统一显示会话提示，没有操作结果时显示默认说明。
    notice_type, notice_text = st.session_state.get("session_notice") or (
        "info", "可以直接提问，也可以切换会话或上传知识文档。"
    )
    show_notice = getattr(st.sidebar, notice_type)
    show_notice(notice_text)
```

### 第一步：设置页面，取得 Agent

`st.set_page_config()` 设置浏览器标签页名称、图标和页面宽度；`st.title()` 在页面中画大标题。`layout="wide"` 表示使用较宽的主区域。

中间的 `st.html()` 只放与顶部排版有关的 CSS 样式：`position: absolute` 让收起按钮所在区域不再单独占一行，`top` 和 `right` 把它放在右上角；`z-index: 1` 让按钮位于标题上层，避免被标题挡住而无法点击。两处 `padding-top` 分别给侧栏内容留一点顶部间距、去掉标题自身额外的顶部留白。底部间距沿用 Streamlit 默认值。这几条样式只作用于侧栏；升级 Streamlit 后需要确认这些页面元素的名称仍然适用。

然后调用 `get_app()`。第一次需要创建模型客户端、连接数据库并同步知识库，之后通常直接取缓存。如果启动失败，`show_error()` 显示原因，`st.stop()` 停止本次脚本，后面的页面不再继续画。

`try / except` 可以读成“先试着做，出了问题就走报错分支”。`except Exception as error` 中的 `error` 是捕获到的异常对象。

### 第二步：准备当前会话号

```python
if "sid" not in st.session_state:
    st.session_state.sid = uuid4().hex
```

`"sid" not in ...` 检查这个键是否还不存在。只有第一次没有会话号时，才生成一个新的。`uuid4().hex` 生成 32 位十六进制编号，供下拉框、读取历史和保存问答共同使用。

下面两种写法读写的是同一项：

```python
st.session_state.sid
st.session_state["sid"]
```

不要每次重跑都重新生成 `sid`，否则点一下按钮就换成另一个空会话了。

### 第三步：画侧边栏，再处理操作

```python
with st.sidebar:
    files, upload_clicked, sync_clicked = render_sidebar(app)
    if upload_clicked:
        handle_upload(app, files)
    if sync_clicked:
        handle_sync(app)
```

`with st.sidebar:` 指定这一块中的页面内容画在侧边栏。函数调用也在这块里执行，所以函数里面的按钮、加载动画、成功提示和错误提示都会放在侧边栏。

`render_sidebar()` 一次返回三个值，左边三个变量按顺序接住。这叫“解包”：文件列表交给 `files`，上传按钮状态交给 `upload_clicked`，同步按钮状态交给 `sync_clicked`。

绘制函数负责报告用户点了什么；`main()` 根据两个布尔值决定是否调用操作处理函数。只是选中文件，还不会入库，必须再点“上传并入库”。

### 第四步：画聊天，再处理新问题

```python
question = render_chat(app)
if question:
    handle_question(app, question)
```

`render_chat()` 先显示已有记录，再画输入框。没有提交时返回 `None`，`if question:` 不成立；提交后返回问题文字，才调用 `handle_question()`。

这一段外面有 `try / except`。绘制主区域或调用 Agent 失败时，显示错误，然后继续执行最后的提示代码。当前这层保护不会回到失败位置继续执行：例如历史记录渲染出错，后面的输入框仍可能没有画出。这是现有异常处理的边界。

### 第五步：统一显示会话提示

```python
notice_type, notice_text = st.session_state.get("session_notice") or (
    "info", "可以直接提问，也可以切换会话或上传知识文档。"
)
show_notice = getattr(st.sidebar, notice_type)
show_notice(notice_text)
```

`session_notice` 保存一个两项元组，例如 `("success", "已切换会话……")`。两个变量分别接住“提示类型”和“提示文字”，比数字下标更容易看懂。

`.get()` 在键不存在时返回 `None`；`or` 表示前面的值为空时，使用后面的默认说明。这样正常页面上总有一条会话提示。

`getattr(st.sidebar, notice_type)` 按名字取方法：

| 类型 | 取到的方法 | 显示效果 |
| --- | --- | --- |
| `"info"` | `st.sidebar.info` | 蓝色说明 |
| `"success"` | `st.sidebar.success` | 绿色成功提示 |
| `"warning"` | `st.sidebar.warning` | 黄色提醒 |

取方法与调用方法分成两行：先交给 `show_notice`，再传入文字执行。类型由本文件中的代码设置。这里明确写了 `st.sidebar`，所以即使代码在函数结尾，提示仍会画到侧边栏底部。

普通重跑会继续显示保存的会话提示；新建、切换或新的问答结果会替换它。上传和同步的处理结果则在各自处理函数中就地显示，不写进 `session_notice`。

## 4. 公共函数：get_app() 和 show_error()

### get_app()：创建一次，后面复用

没有参数，返回一个 `AgentApp` 对象。

```python
@st.cache_resource
def get_app() -> AgentApp:
    return AgentApp(load_settings())
```

嵌套调用从里往外执行：

1. `load_settings()` 读配置、转换类型，创建 `Settings` 时自动检查必填项、范围、地址和端口。
2. `AgentApp(...)` 使用检查通过的配置，创建模型客户端、MySQL 和 Chroma，再同步知识原件。
3. `return` 把应用对象交给调用者。

`@st.cache_resource` 是装饰器，可以理解为给函数加上“复用结果”的规则：第一次执行函数体，后续同样的调用直接返回缓存对象。它适合连接池、模型客户端这样的资源。`st.cache_data` 则用于缓存查询结果等数据，读取时提供副本。

页面普通重跑不会重建这些资源；修改 `.env` 后，需要完整重启应用，才能重新读取配置。

`-> AgentApp` 是返回值类型标注，告诉读者这个函数会返回什么。Python 不会仅凭这段标注自动转换或检查返回值。

### show_error(error)：把异常画成红色提示

参数 `error` 是异常对象；返回 `None`，表示主要作用是显示内容。

```python
logger.exception("页面操作失败")
detail = str(error) if isinstance(error, ValueError) else "请检查开发文档中的配置和网络说明。"
st.error(f"{type(error).__name__}：{detail}")
st.caption(f"详细日志：{LOG_FILE}")
```

`A if 条件 else B` 是写在一行里的条件选择。这里遇到 `ValueError` 时显示它的文字，其他异常使用通用说明。

`isinstance(error, ValueError)` 会把 `ValueError` 的子类也算进去。它只判断异常类型，不能证明错误一定由我们自己抛出。`str(error)` 取错误文字，`type(error).__name__` 取类型名称，例如 `ValueError`、`OperationalError`。

`st.error()` 画红色提示，不会自己中断脚本。要停止执行，需要另行调用 `st.stop()`。所以启动失败后会停下，上传失败后则可以继续画聊天区。


`show_error()` 由 `except` 调用，所以 `logger.exception()` 能附上当前异常的完整堆栈。网页显示简短提示和日志路径，技术细节在 `data/logs/app.log` 与控制台查看。`main()` 先调用 `setup_logging()` 再创建 Agent；日志怎么落盘和轮转，见 [logging_config.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/logging_config.md)。

## 5. 会话回调：new_session() 和 switch_session()

这两个函数都没有参数，返回 `None`。它们由 Streamlit 在控件触发后的页面重跑之前执行。

### new_session()：换一个会话号

```python
st.session_state.sid = uuid4().hex
st.session_state.session_notice = (
    "info", "已创建新会话。完成一轮问答后自动保存。"
)
```

第一行换编号，第二行准备提示。它不删除原聊天，也不创建一条空聊天记录。等这个新会话完成第一轮问答，MySQL 中才有对应记录。

个人画像按用户保存，新旧会话共用它，所以新建会话后，姓名等已保存的个人信息仍然能被使用。

### switch_session()：下拉框已经更新编号，这里补提示

```python
st.session_state.session_notice = (
    "success", "已切换会话，聊天记录显示在主区域。"
)
```

下拉框的 `key="sid"` 已经把选择写进状态，回调只需准备一句提示。重跑时 `render_chat()` 按新的 `sid` 查询历史。

传回调时写 `on_click=new_session`，没有括号：交给 Streamlit 稍后调用。写成 `new_session()` 就会现在立即执行，意思不同。

## 6. 知识库操作：handle_upload() 和 handle_sync()

### handle_upload(app, files)：把选中的文件交给后台

`app` 是缓存的 Agent；`files` 是上传控件返回的文件对象列表；返回 `None`。

```python
uploads = [(file.name, file.getvalue()) for file in files]
with st.spinner("读取文档并建立知识库…"):
    result = app.knowledge.upload(uploads)
st.success(f"已入库 {len(result['files'])} 份文档，知识库共 {result['chunks']} 个切片")
```

按顺序读：

1. 列表推导式逐个取文件，`file.name` 是文件名，`file.getvalue()` 是文件字节。
2. 把它们组成 `(文件名, 字节)` 元组，整理成后台接收的列表。
3. 调用 `app.knowledge.upload()`，校验、保存原件并同步索引。
4. 显示文件数和知识库的切片总数。

`with st.spinner(...)` 表示执行块内代码时显示加载动画，结束后消失。`len(result['files'])` 数上传成功的文件；`result['chunks']` 是全部切片数，不是本次新增数量。

这段有自己的 `try / except`。上传失败时显示原因，之后 `main()` 仍会继续画主聊天区。

### handle_sync(app)：重新扫描已有原件

只有一个参数 `app`，返回 `None`。

```python
with st.spinner("切片与向量化中…"):
    count = app.knowledge.ingest()
st.success(f"知识库已同步：{count} 个切片")
```

适用于直接修改了预置文档或已上传原件的情况。后台重新读取、切片和比对指纹，只向量化新增内容，再清理已不存在的切片。这里返回的也是总数。

同步失败同样交给 `show_error()`，这次操作结束后继续画主区域。

## 7. handle_question(app, question)：调用一轮 Agent

参数 `app` 是应用对象，`question` 是输入框交回的用户文字。成功时会触发页面重跑；失败时异常交给 `main()` 显示。

```python
st.session_state.session_notice = None
with st.spinner("Agent 正在调用模型与工具…"):
    result = app.chat(st.session_state.sid, question)
```

先清空上次的成功提示，再调用 Agent。否则这次失败了，页面上可能还留着上次的“已保存”，让人误以为这次也保存成功。

`app.chat()` 接收当前会话号和原问题，完成检索、模型与工具循环、可选答案检查、记忆提取和保存。返回字典中的 `memory_warning` 决定提示颜色：

```python
if result.get("memory_warning"):
    st.session_state.session_notice = (
        "warning", "本轮问答已自动保存到 MySQL。" + result["memory_warning"]
    )
else:
    st.session_state.session_notice = (
        "success", "本轮问答已自动保存到 MySQL，结构化记忆已检查更新。"
    )
st.rerun()
```

有警告表示问答已保存，但画像更新出了问题，例如提取失败或容量超限。没有警告就准备成功提示。`+` 把两段文字接起来；`.get()` 在键缺失时返回 `None`，空字符串和 `None` 都会让 `if` 不成立。

最后为什么重跑？因为当前页面的历史区是在调用 Agent **之前**画的，还没有这轮新答案。`st.rerun()` 立刻结束当前运行，再从头执行脚本，`render_chat()` 重新查 MySQL，就能画出新问答。

页面不另存第二份聊天列表：后台先保存，页面再读取。会话提示也等下一轮运行到 `main()` 结尾时显示。

## 8. render_sidebar(app)：画控件，返回本轮操作

接收 `app`，返回 `(files, upload_clicked, sync_clicked)`。返回类型 `tuple[list, bool, bool]` 可以读成“一个元组，里面依次是列表、布尔值、布尔值”。

### 先查出历史会话，整理下拉选项

```python
saved_sessions = {row["session_id"]: row for row in app.memory.list_sessions()}
options = [*saved_sessions]
if st.session_state.sid not in saved_sessions:
    options.insert(0, st.session_state.sid)
```

`list_sessions()` 返回最近保存过的会话。字典推导式把它整理成“会话号 → 对应记录”，方便后面用 ID 找轮数、时间。

`[*saved_sessions]` 把字典的键展开成列表。新会话还没存过问答，不在数据库汇总里，所以要额外插到选项最前面，保证下拉框能显示当前会话。

查询失败时会显示错误并 `st.stop()`，因为这时无法正常准备会话选择。

### 会话选项用稳定的 ID，变化信息另外显示

```python
st.selectbox(
    "切换会话（最近 50 个）",
    options=options,
    key="sid",
    on_change=switch_session,
)
saved = saved_sessions.get(st.session_state.sid)
if saved:
    st.caption(f"已保存 {saved['turn_count']} 轮 · 最近保存：{saved['last_saved_at']:%m-%d %H:%M}")
```

| 参数或变量 | 作用 |
| --- | --- |
| 第一个字符串 | 控件上方显示的名称 |
| `options` | 完整的会话 ID；显示的文字与实际值一致 |
| `key="sid"` | 把控件与 `st.session_state.sid` 绑定 |
| `on_change` | 选项改变后，调用 `switch_session` |
| `saved` | 当前会话的统计信息；新会话尚未保存时为 `None` |

轮数、保存时间会变化，所以用下方 `st.caption()` 单独显示。`%m-%d %H:%M` 把日期显示成类似“10-22 00:31”。选项本身始终是同一个完整 ID，不会因为完成一轮聊天改变文字，也不会因为只显示前 12 位而混淆相似编号。

**为什么去掉动态 `format_func`？** 原来新会话显示“当前会话 · …”，保存后变成“… · 1轮 · 时间”。本项目使用的 Streamlit 会拿浏览器提交的显示文字找回实际选项。浏览器如果仍提交旧文字，匹配失败时就可能把旧文字交回来；因为控件绑定 `sid`，后面读历史就把中文、空格和省略号当成了会话 ID，于是校验报错。

因此这里让选项文字保持稳定，把变化信息移出选项。`validate_session_id()` 继续保留，不能通过放宽校验来掩盖页面状态错误。已经处于错误状态的旧浏览器会话，重启应用后重新选择已保存会话即可，聊天记录不需要删除。

### 新建会话按钮的 key 是什么

```python
st.button("新建会话", key="new_session_clicked", on_click=new_session, width="stretch")
```

`key` 是按钮的唯一标识，同时让 Streamlit 管理对应的按钮状态。`st.session_state["new_session_clicked"]` 表示本轮是否点击：初始为 `False`，点击触发的这一轮为 `True`，本轮结束后恢复为 `False`。这是按钮控件的行为，不需要手动赋值。

这个状态与 `sid` 分别保存。按钮状态恢复为 `False` 不会清掉新会话号。页面通过回调执行动作，不主动读取按钮状态；测试则用这个 `key` 找到按钮。`width="stretch"` 让按钮铺满所在区域的宽度。

### 知识库管理：折叠内容，按钮并排

```python
with st.expander("知识库管理", expanded=False):
    files = st.file_uploader(
        "上传知识文档", type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True, max_upload_size=10,
    )
    upload_column, sync_column = st.columns(2)
    upload_clicked = upload_column.button(
        "上传并入库", disabled=not files, width="stretch",
    )
    sync_clicked = sync_column.button("同步知识库", width="stretch")
```

`st.expander()` 像一个可打开的抽屉，`expanded=False` 设置默认收起。它只隐藏内容，**不会阻止里面的 Python 代码执行**，也不是后台懒加载机制。

上传控件参数的意思：

| 参数 | 含义 |
| --- | --- |
| `type=[...]` | 允许选择这些扩展名 |
| `accept_multiple_files=True` | 可以选多份文件，返回文件对象列表 |
| `max_upload_size=10` | 每份文件最多 10 MB |

`st.columns(2)` 创建两列，分别由两个变量接住；调用 `upload_column.button()` 就把按钮画到第一列。两个按钮并排，减少侧边栏高度。

`disabled=not files` 表示没有选中文件时，上传按钮置灰。`upload_clicked`、`sync_clicked` 是普通局部变量，接收按钮本轮返回的 `True` 或 `False`，交给 `main()` 判断。

### 运行信息：直接显示

这里先用 `st.markdown("**运行信息**")` 画加粗的小标题，再通过四次 `st.caption()` 显示用户、问答模型、向量模型和数据库名称。没有套在 `st.expander()` 中，所以始终直接显示。`caption` 是灰色小字，适合辅助信息。

函数最后返回三个值。它没有在画上传、同步按钮时执行后台操作，处理入口在 `main()` 的两个 `if` 中，一眼就能找到。

## 9. render_chat(app)：显示历史，交回新问题

接收 `app`，返回 `str | None`。竖线表示“可能是字符串，也可能是 `None`”。按页面从上往下看：会话说明、画像、历史问答与本轮用量、当前用户累计、输入框。

### 会话说明

用 `st.caption()` 显示当前完整会话号、累计摘要与未压缩原文说明、页面显示最近 50 轮，以及几个示例问题。模型读取累计摘要及覆盖位置之后的全部问答，与页面显示最近 50 轮分别处理。有摘要时用一行说明显示已覆盖的真实对话 ID。

### 个人画像

```python
today = datetime.now().astimezone().date()
profile = app.memory.profile()
render_profile(profile, today)
```

`render_profile()` 把“展开面板 + 有画像画 JSON / 没画像给提示”这段重复渲染收在一处。`profile()` 不传会话号，因为画像按用户共享；`today` 用来现算有效期。

聊天区有两个调用位置：历史记录之前调用一次，保证还没聊天时也能看到画像；每条助手消息之后再调用一次，方便对照“这轮回答时”的最新画像。两处展示的是同一份当前画像，不是历史快照。

外层字典推导式逐个处理字段，`{**fact, "status": ...}` 先展开原字段的内容，再补上状态，形成新字典，不修改原画像。状态由 `fact_status()` 现算，不写回数据库。`st.json()` 把结果画成可查看的结构。

空字典在 `if` 判断里算“假”，所以没有画像时显示蓝色说明。遗忘只弃用画像，原始聊天仍然保留。

### 历史问答和轨迹

```python
for row in app.memory.recent(st.session_state.sid):
    with st.chat_message("user"):
        st.markdown(row["user_text"])
    with st.chat_message("assistant"):
        st.markdown(row["assistant_text"])
        trace = json.loads(row["trace_json"])
```

`recent()` 按当前会话查询最近记录。`st.chat_message()` 创建聊天气泡，参数决定是谁说话；`st.markdown()` 画正文，能显示列表、加粗、代码块等 Markdown 格式。

`trace_json` 在数据库里是文本，`json.loads()` 把它还原成 Python 列表。源码会跳过损坏的 JSON 和非字典元素。用量直接读取独立的 `usage_json`，工具与反思仍从轨迹里分开显示：

```python
render_usage((row.get("usage_json") or {}).get("current"))
tool_trace = [item for item in trace if item.get("kind") != "reflection"]
```

旧工具记录没有 `kind` 时，`.get()` 返回 `None`，仍归入工具轨迹。没有工具记录就不画对应抽屉。

接着遍历 `kind == "reflection"` 的记录，显示“答案检查”：

- `review_failed`、`revision_failed` 显示黄色提醒；
- 其他状态显示检查结果说明；
- `issues` 非空才展示问题清单；
- 展示检查前草稿，方便与最终答案对照。

`usage_json` 由数据库读取时就已经是字典；没有记录时是 `None`，显示“未记录”。这里不再从工具轨迹里查找用量。这些内容从数据库读取，这里不再调用模型；能解析的反思和用量条目仍需符合程序保存的字段结构。

### `render_usage()`：两行显示六项数字

这个函数接收 `stats` 和显示前缀 `prefix`。每条回答默认用“本轮”，底部传“累计”，所以不用维护两套展示代码。

第一行显示输入、输出、总 token，用千位逗号方便读数。第二行显示缓存命中率、推理 token 占比和处理耗时。缓存率用 `cached_tokens / input_tokens`，推理占比用 `reasoning_tokens / output_tokens`；分母为零显示“—”，字段为 `None` 显示“未提供”，明确返回 0 才显示零。

这些只是接口返回的数量和时间，不是模型的思考内容，也不额外调用模型。具体统计过程见 [usage.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/usage.md)。

### 底部累计：直接显示当前会话最后一行的总数

`rows` 就是上面画聊天时已经读取的最近 50 轮，不另查一遍：

```python
latest_usage = (rows[-1].get("usage_json") or {}) if rows else {}
total = latest_usage.get("total")
render_usage(total, prefix="累计")
```

每轮保存时已经算好 `total`，所以这里只负责显示。即使会话超过 50 轮，最后一行仍带着从第一轮开始的累计。统计轮数直接取 `total["turns"]`。

刷新页面不会重新累加；新会话从零开始；切回旧会话就恢复它的累计。没有统计显示“未记录”，接口缺少字段显示“未提供”。累计比例依然用累计缓存数除累计输入数、累计推理数除累计输出数，不平均百分比。

耗时显示 `chat_seconds`，名称是“处理耗时”：从进入 `chat()` 到保存前，包括检索、工具、反思、记忆提取和摘要，不包含最后保存和网页绘制。token 不包含 Embedding 或未保存的失败问答。

### 输入框

```python
return st.chat_input("输入问题")
```

在这里创建的聊天输入框固定在主页面底部。未提交返回 `None`，提交后返回文字，直接交回 `main()`。所以绘制函数到这里结束，调用 Agent 的细节在 `handle_question()` 中。

## 10. 把一次提问从头走完

```text
用户提交“我叫小明”
    → Streamlit 重跑脚本，调用 main()
    → get_app() 复用缓存
    → 保留当前 sid
    → render_sidebar() 画控件，上传和同步都没点击
    → render_chat() 读历史并画页面，返回“我叫小明”
    → handle_question() 清掉旧提示，调用 app.chat()
        → 后台生成回答、提取画像、保存问答
        → 准备成功或警告提示
        → st.rerun()，立即结束这次运行
    → 再次进入 main()
    → render_chat() 读到刚保存的问答，画出新气泡
    → 输入框这次没有新提交，返回 None，不再调用 Agent
    → main() 结尾显示刚准备好的提示
```

切换会话时，先由控件更新 `sid`、执行回调，再重跑页面读取新会话。上传时，选择文件触发重跑只会画页面；点击上传按钮才调用 `handle_upload()`，普通重跑不会重复入库。

## 11. 开头的导入与常用语法速查

| 名字或写法 | 在这个文件里的用途 |
| --- | --- |
| 模块开头的三引号文字 | 说明文件职责，Python 会保存为模块文档字符串 |
| `import json` | 把数据库里的工具轨迹文本还原为列表 |
| `from datetime import datetime` | 获取日期，判断画像状态 |
| `from uuid import uuid4` | 生成新会话编号 |
| `import streamlit as st` | 给 Streamlit 起一个短名字 |
| `AgentApp` | 创建后台应用，执行问答或知识库操作 |
| `load_settings` | 读取并检查配置 |
| `fact_status` | 算出一条画像当前是否有效 |
| `def ... -> None` | 定义函数，主要执行动作，不返回业务数据 |
| `return ...` | 把结果交给调用者，结束当前函数 |
| `if question:` | 有新提交才处理；空字符串、`None` 都算假 |
| `.get("key")` | 键不存在时返回 `None` |
| `f"{变量}"` | 把变量值填进字符串 |
| `for ... in ...` | 逐个处理文件、会话或聊天记录 |
| `[... for ...]` / `{... for ...}` | 把遍历和整理列表、字典写在一起 |
| `with ...` | 进入指定区域或执行范围，退出后恢复 |
| `st.stop()` | 停止这次脚本执行 |
| `st.rerun()` | 停止这次执行，并马上开始下一次重跑 |

阅读这个文件时，先沿着 `main()` 看“先做什么、后做什么”，再进入对应函数查看“具体怎样做”。看到 `app.chat()` 后，要继续理解模型、工具和记忆的业务过程，就转到 [agent.py 的讲解](D:/study/AI/ai-agent-demo-study/docs/code-guide/agent.md)。
