# logging_config.py：报错时留下一本本地记录

对应源码：[agent_demo/logging_config.py](D:/study/AI/ai-agent-demo-study/agent_demo/logging_config.py)。这个模块只做一件事：**把日志同时写到文件和控制台**。用的是 Python 标准库 `logging`，不需要安装新包，也不需要注册服务账号。

## 1. 先看东西放哪里

```text
data/logs/
  app.log       ← 正在写的日志
  app.log.1     ← 最近一份旧日志
  app.log.2
  app.log.3     ← 最多保留三份备份
```

`LOG_FILE = ROOT / "data" / "logs" / "app.log"` 是完整路径。从 PyCharm 启动或在其他目录执行，都会写到同一处。

日志是普通 UTF-8 文本，PyCharm、记事本、VS Code 都能打开。也可以在项目目录用 PowerShell 看最后的内容：

```powershell
Get-Content ./data/logs/app.log -Tail 100 -Encoding UTF8
# 加 -Wait，可以一边运行项目，一边看新日志。
```

## 2. `setup_logging()`：先把记录工具准备好

函数不接收参数，也不返回业务结果。网页 `main()` 最开始调用它，命令行评测入口也调用它，所以连初始化失败也能留下记录。

### 第一步：拿到项目自己的记录本

```python
logger = logging.getLogger("agent_demo")
if logger.handlers:
    return
```

`getLogger("agent_demo")` 拿到这个名字对应的日志对象。其他模块使用 `agent_demo.agent`、`agent_demo.reflection` 等名字，它们的日志会交给 `agent_demo` 统一输出。

`handlers` 是已经装好的输出通道。Streamlit 每点一次按钮都会重跑页面，已经装好就直接返回，不再装一遍，否则同一条报错可能打印好几次。

网页脚本运行时名字是 `__main__`，所以它明确用 `logging.getLogger("agent_demo.app")`，让网页的日志也归进这本记录。

### 第二步：创建日志目录和文件通道

```python
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
```

| 参数 | 大白话 |
| --- | --- |
| `LOG_FILE` | 要写哪个文件 |
| `maxBytes` | 一份文件约写到 5 MB 就换一份 |
| `backupCount=3` | 最多保留三份旧文件，更早的会被替换 |
| `encoding="utf-8"` | 中文用 UTF-8 存，方便直接打开 |

`mkdir()` 会补齐缺失的目录，目录已有时不报错。`RotatingFileHandler` 默认继续追加，不会每次启动都把旧日志清空。

### 第三步：再接到控制台

```python
console_handler = logging.StreamHandler()
```

它把同一条记录输出到控制台，例如 PyCharm 的运行窗口。文件负责留存，控制台方便当场查看。

### 第四步：统一每行长什么样

```python
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s")
```

一条记录大致长这样：

```text
2026-10-21 23:53:04,623 ERROR agent_demo.app:32 页面操作失败
Traceback (most recent call last):
  ...中间是调用过的文件、行号和函数...
chromadb.errors.InternalError: error returned from database: (code: 14) unable to open database file
```

前半行告诉你什么时候、什么级别、哪个模块哪一行写的日志。后面的堆栈说明异常一路从哪里传上来，最后一行是具体异常内容。

`for handler in (...)` 把同一个格式交给文件和控制台两个通道，再 `addHandler()` 安装到项目记录本上。

### 第五步：规定记录范围

```python
logger.setLevel(logging.INFO)
logger.propagate = False
```

`INFO` 及更严重的级别会输出。本项目启动步骤用 `info()`，报错用 `exception()`。不主动记录每条提示词、画像、工具正文，也不开启第三方 HTTP 的详细调试日志。

`propagate=False` 表示本项目记录到这里就完成，避免再经过 Streamlit 等上层配置输出一遍。

## 3. 其他文件怎样写日志

一般模块顶部先写：

```python
import logging
logger = logging.getLogger(__name__)
```

`__name__` 就是当前模块名，自动把来源带到日志里，不需要每次手填文件名。

正常步骤：

```python
logger.info("初始化 MySQL 记忆库")
```

捕获异常的 `except` 里面：

```python
except Exception:
    logger.exception("答案检查失败，沿用草稿")
```

`exception()` 会自动附上当前异常的完整堆栈。只写 `logger.error(str(error))` 通常只有一行报错，看不到出错函数和调用关系。

**`exception()` 要在处理异常时使用。** 这里的网页 `show_error()` 也是由 `except` 分支调用，当前异常仍然存在，因此能记录完整堆栈。

## 4. 当前哪些地方已经接上

| 位置 | 记录什么 |
| --- | --- |
| `app.py:main()` | 最开始打开日志，初始化失败也能记录 |
| `app.py:show_error()` | 页面捕获的启动、会话读取、聊天、上传、同步等异常；页面同时提示日志路径 |
| `agent.py:AgentApp.__init__()` | 创建模型客户端、连接 MySQL、打开 Chroma、开始与完成知识同步 |
| `agent.py:AgentApp.chat()` | 画像合并失败后退回只存问答的异常 |
| `reflection.py:reflect_answer()` | 检查或修订失败、沿用草稿时的完整原因 |
| `structured_memory.py:extract_changes()` | 记忆提取失败、画像不更新时的完整原因 |
| `evaluate.py` | 单题异常和评测入口异常 |

网页继续显示简短提示，完整技术细节留在日志里。日志包含异常原文，分享给别人前检查其中是否有需要遮掉的信息。

## 5. 这次 Chroma 报错说明什么

调用链是 `app.py:get_app()` → `AgentApp.__init__()` → `KnowledgeBase.__init__()` → Chroma 的 SQLite 初始化。

本机对比结果：MySQL 正常，原 SQLite 的完整性检查正常，索引目录可写；默认 Windows 临时目录下 Chroma 初始化失败，当前进程改用项目内的可写临时目录后成功。

因此 `rag.py` 在 Windows 下先创建 `data/tmp`，再让当前进程的 `TEMP`、`TMP` 指向它，随后才创建 Chroma。临时文件目录和持久化索引目录不是同一个概念：前者是工作中暂时使用的位置，后者是保存资料的位置。

修改后需要**停止 PyCharm 中的旧进程，再重新运行**。Chroma 首次初始化失败后可能在进程内留下失效对象，刷新页面不能保证清掉它。

## 6. 这个简版做到哪里

这是本机教学项目的文件日志，按单进程使用，不增加日志数据库、追踪服务、队列或自定义日志类。日志位置固定为项目根目录的 `data/logs`，评测也使用这里。

要清理日志或整个 `data/`，先停止程序，再删除文件。新启动时会自动创建日志目录。大规模多进程部署时再考虑集中收集，这次不增加相应代码。

标准库说明：[Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)、[RotatingFileHandler](https://docs.python.org/3/library/logging.handlers.html#rotatingfilehandler)。
