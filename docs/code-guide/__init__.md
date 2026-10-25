# __init__.py：工具箱的封条——先搞懂"包"和"导入"

对应源码：[agent_demo/__init__.py](D:/study/AI/ai-agent-demo-study/agent_demo/__init__.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

`agent_demo` 文件夹就像一个**工具箱**：`config.py` 是配置工具、`memory.py` 是记账工具……而 `__init__.py` 就是**箱子上的封条和清单**——它告诉 Python"这个文件夹是个正经工具箱，可以按名字进来拿东西"，顺便写了一段"开箱说明"。

本项目的这个文件里**一行逻辑代码都没有**，只有一段三引号说明文字。不需要在这里找任何功能，它存在的意义是让 `from agent_demo.xxx import yyy` 这种写法能工作。

## 1. 先补三个小概念

| 名字 | 白话解释 | 本项目里的例子 |
| --- | --- | --- |
| 模块 | 一个可以导入的 Python 文件 | `config.py` |
| 包 | 放了 `__init__.py` 的目录，也就是"工具箱" | `agent_demo` |
| 导入 | 按名字找到代码，拿来用 | `from agent_demo.config import Settings` |

导入**不是**"把别人文件的文字复制过来"，更接近"顺着门牌号找到那个柜子，把指定的一样东西搬过来"。

## 2. 文件开头那段三引号文字会执行吗

那段文字叫**模块说明（docstring）**，Python 把它存进 `__doc__` 供人查看：

```python
import agent_demo
print(agent_demo.__doc__)   # 打印出那段说明文字
```

关键认知：**说明里写了 `memory`、`tools` 这些词，Python 不会因此创建记忆对象或调用工具**。它只是文字，就像工具箱封条上印的"内含锤子、扳手"不等于箱子里自动动起来了。

**"提到一个名字"和"真的执行它"是两回事**——这句话后面读每个文件都用得上。

## 3. `from ... import ...`：一步步找东西

```python
from agent_demo.config import Settings
```

分三步读：

1. 找到 `agent_demo` 这个包（顺带：第一次进箱会执行 `__init__.py` 的顶层代码，本项目里就是"读了段说明，啥也没干"）；
2. 找到包里的 `config.py`（不用写 `.py` 后缀，点号表示往里一层）；
3. 取出里面定义的 `Settings` 类名，当前文件从此可以用它。

第一次加载 `config.py` 时，Python 会从头执行它的顶层代码——遇到 `class Settings:` 就创建这个类。**但类里写的 `def validate()` 只是"把工具挂上架子"，函数体一行都没跑**。不调用，永远不执行。

同一进程里第二次导入时，Python 发现"这个模块已经加载过了"，直接复用，不会重新执行一遍。

## 4. 两个 `__init__` 别认错

这是新手最容易眼花的一对，名字一样、完全是两码事：

| 写法 | 是什么 | 什么时候跑 |
| --- | --- | --- |
| `__init__.py` | 工具箱的封条文件 | 第一次导入这个包时 |
| `def __init__(self, settings)` | 类里的"开机自检"方法 | 每次创建这个类的对象时 |

```python
from agent_demo.config import Settings   # 这一步：找到 Settings 这个"图纸"
settings = Settings(                    # 这一步：创建对象，自动赋值并执行 __post_init__
    llm_api_key="test-key",
    llm_base_url="https://example.invalid/compatible-mode/v1",
    mysql_user="test-user",
    mysql_password="test-password",
    summary_keep_turns=2,  # 仅在压缩时优先保留最近两轮原文。
)
```

创建 `Settings` 时会检查必填项和格式，所以练习也要填齐。上面全是虚构值，只用于理解创建对象的过程；检查不会连接真实服务。后续使用 `AgentApp(settings)` 则会连 MySQL、打开知识库、可能给百炼发请求，不能拿这份虚构配置启动它。

## 5. 那聊天网页是从哪冒出来的？

不是从 `__init__.py`。路线是：

```text
Streamlit 运行根目录的 app.py
    → app.py 导入 agent_demo 里的各模块
    → 读配置、创建 AgentApp
    → 网页开始收问题
```

单独右键运行 `__init__.py` 什么也不会发生——封条自己不是机器。启动方法看 [项目 README](D:/study/AI/ai-agent-demo-study/开发文档.md)。

## 本篇要带走的三句话

1. `__init__.py` = "这个目录是包"的标记，本项目里内容只有一段说明；
2. 导入是"找到并使用名字"，不会触发函数执行；
3. `__init__.py`（文件）和 `def __init__`（方法）除了名字长得像，没有半毛钱关系。

继续读：[config.py：传达室大爷——配置单](D:/study/AI/ai-agent-demo-study/docs/code-guide/config.md)。
