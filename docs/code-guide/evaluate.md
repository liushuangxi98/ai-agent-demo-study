# evaluate.py：给整个项目做一次"真枪实弹的体检"

对应源码：[evaluate.py](D:/study/AI/ai-agent-demo-study/evaluate.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

前面各篇讲的都是"这个项目怎么工作"。`evaluate.py` 回答的是另一个问题：

> **它到底工作得好不好？**

它是一个**自动体检脚本**：程序自己提一组题问自己的 Agent，看答得对不对、有没有真的调对工具、数据有没有真的存进数据库，最后打一份体检报告出来。默认 13 项；开启 Reflection 时另外加 2 项检查，共 15 项。

跑法就一条命令（在项目根目录）：

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

---

## 0. 开读前先分清："单元测试"和"真实评测"是两回事

这是理解整个文件最重要的一条。项目里其实有**两套**测试，各管一摊，谁也不能替代谁：

| | `tests/` 下的 pytest | `evaluate.py` |
| --- | --- | --- |
| 跑法 | `python -m pytest -q` | `python evaluate.py` |
| 连不连百炼 | **不连**，全部用假的模拟对象 | **真连**，真的调用模型和向量接口 |
| 连不连 MySQL | **不连**（只有 `test_memory_orm.py` 用临时内存库） | **真连**，真的读写数据库 |
| 查不查 Chroma | 不查 | 真的向量化、真的检索 |
| 跑一次的成本 | 0 元、几秒钟 | **要花钱、要等**（每次提问都要调模型） |
| 能测出什么 | 代码逻辑对不对（函数输入输出、格式校验、事务回滚） | **整条链路串起来到底能不能用** |

为什么两套都要？

- `pytest` 快、免费、随时跑，但它测的是"**零件**"。零件全合格，装成整机还是可能不转——比如密钥配错了、模型服务挂了、MySQL 连不上，这些 pytest 一个都发现不了（因为它压根不连）。
- `evaluate.py` 慢、要花钱，但它测的是"**整机**"。它模拟的是一个**真实用户从零开始用这个系统**的过程——提问、记住名字、换个会话、更正、遗忘、上传文档、再提问。

所以一句话总结这个文件的定位：**它是"用户视角的验收测试"，而不是"程序员视角的单元测试"。**

> **前提条件**：跑它之前，`.env` 必须配好、MySQL 必须起着、模型服务必须能通。缺一样它就会直接失败退出——这本身就是它想告诉你的信息。

---

## 1. 模块说明

```python
"""真实效果评测：使用配置的百炼 API、MySQL 和 Chroma，无运行模式开关。"""
```

还是**模块文档字符串**（`app.py` 第 1 节讲过，会被 Python 保留，能用 `__doc__` 读）。

这句话里的每个词都在强调同一件事——**"这是真的"**：

| 词 | 含义 |
| --- | --- |
| `真实效果评测` | 不是模拟，是实际效果 |
| `配置的百炼 API` | 用你 `.env` 里填的那个密钥真调 |
| `MySQL 和 Chroma` | 真写数据库、真写向量库 |
| **`无运行模式开关`** | **这句最值得琢磨** |

**"无运行模式开关"是什么意思？**

很多项目会有个开关，比如 `MOCK=true` 就走假数据、`MOCK=false` 就走真服务。这个项目**故意不设这种开关**：

- 想跑假测试 → 去 `tests/` 用 pytest；
- 想跑真评测 → 跑 `evaluate.py`。

**两条路彻底分开，不留模糊地带。** 好处是：你永远不会遇到"我以为在测真东西，其实开关忘关了"这种事。

---

## 2. 导入

```python
import json
import logging
from dataclasses import replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent_demo.agent import AgentApp, SYSTEM_PROMPT
from agent_demo.config import ROOT, load_settings
from agent_demo.logging_config import setup_logging
from agent_demo.memory import ConversationMemory
```

还是三组：标准库 → 空行 → 自己项目的代码。**这里没有第三方库**（不需要 Streamlit，因为这是个命令行脚本）。

逐个说：

### 2.1 `import json` —— 写报告用

文件和 `app.py` 一样是标准库，但**用法刚好相反**：

| | 用途 |
| --- | --- |
| `app.py` 里 | `json.loads(...)` 把数据库里的文本**读**成 Python 对象 |
| `evaluate.py` 里 | `json.dumps(...)` 把 Python 字典**写**成 JSON 文本存进文件 |

正好对应第 2.1 节说的那句口诀：**`loads` 是装进来，`dumps` 是倒出去。**

### 2.2 `from dataclasses import replace` —— 本文件最关键的技巧

这一行是整个 `evaluate.py` 的**灵魂**。先搞懂它，后面两处关键代码才看得懂。

**背景**：`agent_demo/config.py` 里的 `Settings` 是这么定义的：

```python
@dataclass(frozen=True)
class Settings:
    llm_api_key: str = ""
    ...
    user_id: str = "local_user"
    data_dir: Path = ROOT / "data"
```

`frozen=True` 的意思是"**冻结的、创建之后不许改**"。这是刻意的设计——配置就该一次定好，不该被到处乱改。所以下面这种写法**会直接报错**：

```python
settings = load_settings()
settings.user_id = "eval_xxx"      # ❌ FrozenInstanceError，冻结对象改不了
```

那评测要换一个用户 ID 怎么办？**不能改原件，就照着原件"复印一份、改掉一格"。**`replace()` 干的就是这个：

```python
new_settings = replace(settings, user_id="eval_xxx")
```

读法是：**"拿 `settings` 当模板，把 `user_id` 这一个字段换成新值，其他字段全部照抄，返回一个新对象。"**

关键点：**原对象一个字没动**。这就是"冻结"的代价和收益——代价是不能就地改，收益是**任何地方拿到的 `Settings` 都是可靠的、不会被别人偷偷改掉**。

> 这个思路和 `app.py` 第 10.2 节的 `{**fact, "status": ...}` 是同一个哲学：**需要变就造一份新的，绝不原地修改。**

### 2.3 `from datetime import datetime` —— 只用来"取个年份"

和 `app.py` 一样是"模块名撞类名"。这里只用在**一道题里**：

```python
[str(datetime.now().year)]
```

意思是"取此刻本机时间的**年份**（比如 `2026`）"。见第 5.2 节。

### 2.4 `from typing import Any` —— 类型标注里的"随便什么"

`Any` 是**类型标注里的"万能牌"**，意思是"**这里什么类型都可能**"。

它只出现在返回值的标注里，比如：

```python
def run_evaluation(app: AgentApp, run_id: str) -> dict[str, Any]:
```

读法是"返回一个字典，**键是字符串，值是什么类型都行**"。

为什么这里必须用 `Any`？看返回的那个字典长什么样：

```python
{
    "passed": 13,                     # 整数
    "total": 13,                      # 整数
    "results": [                      # 列表
        {"case": "...", "passed": True, "answer": "...", "trace": [...], "memory_warning": ""},
        {"case": "...", "passed": False, "error": "ValueError"},   # ← 注意这个键完全不一样
        ...
    ],
}
```

`results` 里每一项的**结构都不统一**——成功的项有 `answer`/`trace`/`memory_warning`，失败的项只有一个 `error`。而且 `trace` 里又是列表套字典。

硬要给这种"自由发挥"的数据标一个精确类型，会写得又长又没意义。所以这里坦率地写 `Any`——**该模糊的地方就模糊**。

> 顺带认识几个常见的 typing 工具，你都在这份文档里见过：`list[str]`（字符串列表）、`str | None`（字符串或空）、`dict[str, Any]`（值为任意类型的字典）。

### 2.5 `from uuid import uuid4` —— 给每次评测发个编号

`app.py` 里也用了它。这里在 `main()` 里生成一次 `run_id`，后续所有评测会话都用这个前缀派生，不再单独生成 UUID。

作用见第 4 节——这是"**互不干扰**"的基础。

### 2.6 项目内的导入

| 导入 | 用途 |
| --- | --- |
| `AgentApp` | 整个 Agent 应用（就是 `app.py` 里用的那个类） |
| `SYSTEM_PROMPT` | 原问答的行为规则，开启反思评测时作为检查依据 |
| `ROOT` | 项目根目录的 `Path`（`config.py` 里算好的） |
| `load_settings` | 读 `.env` |
| `setup_logging` | 命令行入口先配置日志，异常堆栈写入 `data/logs/app.log` |
| `ConversationMemory` | **单独**连一次数据库，用于独立复核（见第 7 节） |

**注意 `AgentApp` 这个导入**：`evaluate.py` 复用的就是网页用的那同一个类。**评测测的是网页用户真正会用的那个东西**，不是另一个仿制品。这是这份评测有价值的根本原因。

`ROOT` 的引入也值得留意——它不是硬写路径，而是拿 `config.py` 里算好的根目录。这样**不管在哪个目录下敲命令**，报告都会落到正确的地方。

---

## 3. `run_evaluation()` 的"门面"

```python
def run_evaluation(app: AgentApp, run_id: str) -> dict[str, Any]:
    """参数：独立评测应用、运行 ID；返回通过数、总数和逐题结果。"""
    results = []
```

**两个参数各管一件事**：

| 参数 | 是谁给的 | 干嘛用 |
| --- | --- | --- |
| `app` | `main()` 建好后传进来 | 被评测的对象 |
| `run_id` | `main()` 生成的 12 位随机串 | 给这次评测的所有会话起名字，防止和上次撞车 |

**为什么不在这里自己建 `app`，而要"外面建好传进来"？**

因为**建 `app` 很重**——要连 MySQL、要建 Chroma 并同步文档。谁建谁负责关，`main()` 才能用 `try/finally` 保证它被关掉。**创建资源和释放资源写在同一个地方**，这是很干净的写法。

**最后一行 `results = []`**：准备一个**空清单**，用来一条条记录每道题的结果。整个 `run_evaluation` 的核心动作就是"**不停往这个清单里追加结果**"，最后在函数末尾数一数。

注意这个变量**在函数体最外层定义**，但被里面那个嵌套函数往里写——见下一节。

---

## 4. `ask()` —— 嵌在里面的"发题助手"

这是全文件最核心的一段。它是个**定义在函数里面的函数**（叫嵌套函数）。

```python
def ask(name: str, session: str, question: str, words: list[str],
        expected_tool: str | None = None) -> dict[str, Any] | None:
```

### 4.1 五个参数

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `name` | `str` | 这道题的**名字**（写到报告里给人看） |
| `session` | `str` | 用**哪个会话**提问 |
| `question` | `str` | 问什么 |
| `words` | `list[str]` | 答案里**必须包含**的词（判定对错的依据） |
| `expected_tool` | `str \| None = None` | 期望调用的工具名，**省略则不检查工具** |

**`expected_tool: str | None = None` 这个写法要拆开读：**

| 片段 | 意思 |
| --- | --- |
| `str \| None` | 类型是"字符串 **或** 什么都没有"（`\|` 是"或者"） |
| `= None` | **默认值**：调用时不传，它就是 `None` |

所以 `ask("短期记忆", sid, "我叫什么名字？", ["小明"])` 这种只传四个参数的调用是合法的——`expected_tool` 自动取 `None`（表示"这道题不管它调了什么工具"）。

**返回值 `dict[str, Any] | None`** 同理："成功时给一个字典，异常时给 `None`"。

### 4.2 为什么要把函数**写在** `run_evaluation` 肚子里？

这又是一次**闭包**（你在 `tools.py` 那篇见过）。

`ask()` 身体里用到了两个"外面来的东西"：

- `app` —— `run_evaluation` 的参数；
- `results` —— `run_evaluation` 里的那个清单。

如果 `ask` 定义在外面（模块级），这两个名字它就**看不见**，只能当成参数一路传进去：

```python
# 如果写在外面，每次调用都得这样
ask(app, results, "短期记忆", sid, "我叫什么名字？", ["小明"], None)
```

那样每次都要多传两个"基础设施"参数，**噪音大、还容易漏**。写在肚子里就直接用，调用处干净得多：

```python
ask("短期记忆", sid, "我叫什么名字？", ["小明"])
```

> **闭包的判断标准**：函数体里用到了"自己参数之外、外层函数里定义的名字"。`ask` 用了 `app` 和 `results`，所以它是闭包。

### 4.3 `try` 分支：正常发一道题

```python
try:
    result = app.chat(session, question)
    answer_ok = all(word in result["answer"] for word in words)
    used_tools = [item["tool"] for item in result["trace"] if item.get("kind") != "reflection"]
    tool_ok = expected_tool is None or expected_tool in used_tools
```

**第一行：真的去问**

`app.chat(session, question)` —— 就是网页上点回车时调的那个方法。**这一步会真的花掉一次模型调用**（严格说是一到多次：工具循环 + 记忆提取）。

返回的 `result` 是个字典，四个键：`answer`（回答文字）、`trace`（工具轨迹）、`memory_warning`（记忆环节的警告）和 `usage`（本轮与会话累计用量）。

**第二行：答案对不对？**

```python
answer_ok = all(word in result["answer"] for word in words)
```

这一行有三个语法要认：

**① `word in 字符串`** —— 子串判断。"`小明` 在不在这段回答里"，返回 `True`/`False`。

**② `for word in words`** —— 挨个遍历要求的词。

**③ 括号里那整个东西是"生成器表达式"**

```python
(word in result["answer"] for word in words)
```

它和"列表推导式"长得很像（`[x for x in ...]`），区别是**用圆括号**。效果上：

| | 列表推导式 `[...]` | 生成器表达式 `(...)` |
| --- | --- | --- |
| 产出 | 立刻造出一整个**列表** | 造一个"**惰性**的流水线"，用的时候才算 |
| 内存 | 全部算完先存起来 | 算一个用一个 |
| 这里用谁 | — | **用的是生成器**，因为下一层 `all()` 只关心"是不是全对" |

**④ `all(可迭代对象)`** —— Python 内置函数，意思是"**里面是不是每一项都为真**"。

所以整行读作：**"要求的每个词，是不是都出现在回答里。"**

举个具体例子：`words = ["7", "20%", "course.md#"]`，三道都命中才算对。**只要有一个没出现，就是 `False`。**

> 顺带认识它的兄弟 `any(...)`（任意一项为真即可）。这两个是判断类代码里出现频率最高的内置函数之一。

**这里用"关键词包含"判对错，是有局限的，值得知道**：它不知道**同义表达**。假如模型的回答是"退款须在**七天**内…"，判分就会认为失败（因为回答里没有数字 `7`）——**模型答对了，却被判错**。

这是"关键词断言"天生的短板，属于**可以接受但必须知道**的取舍：

- 好处：简单、免费、不受模型波动影响；
- 代价：可能误判，所以报告里同时留下了 `answer` 原文和 `trace`，**让你能人工复核**。

所以文档里才提醒："**结合答案、工具证据和 `memory_warning` 判断结果**"，别只看那个红叉。

**第三行：用了哪些工具？**

```python
used_tools = [item["tool"] for item in result["trace"] if item.get("kind") != "reflection"]
```

这是**列表推导式**（你在 `app.py` 那篇第 9.2 节见过字典版的）。

`result["trace"]` 长得像：

```python
[
  {"tool": "knowledge_search", "args": {...}, "result": "...", "trigger": "automatic"},
  {"tool": "calculator", "args": {"expression": "(12+8)*3"}, "result": "计算结果：60"},
]
```

推导式做的事是"**从每一条记录里把工具名抠出来，拼成一个纯名字的列表**"：

```python
["knowledge_search", "calculator"]
```

**为什么要专门抠出来？** 因为接下来只需要回答一个问题——"**期望的那个工具在这堆名字里吗**"，不需要那些参数和结果的细节。

**第四行：工具对不对？**

```python
tool_ok = expected_tool is None or expected_tool in used_tools
```

这是个**短路或**（`or`）的经典用法，读法要按语义读：

**"没要求检查工具（是 `None`）** —— 那就直接算通过；**否则**才去看期望的工具名在不在实际用到的里面。"

也就是说：**不指定** `expected_tool` = **放弃这项检查**，而不是"必须一个工具都不用"。

**顺带说一个必须记住的坑：这里用的是 `is None`，不是 `== None`。**

| 写法 | 含义 |
| --- | --- |
| `x is None` | **推荐**。"是不是就是那个独一无二的空对象" |
| `x == None` | 能跑，但**不推荐** |

为什么？因为 `==` 可以被自定义（有些对象的 `==` 会对很多东西都返回 `True`），而 `is` 比较的是"**是不是同一个对象**"。`None` 在 Python 里全局只有一个，所以只能用 `is` 判断，这也是 PEP 8 的明确要求。

**判断"空"一律用 `is None` / `is not None`**，这是个看起来很小但很见功力的习惯。

**追加结果：**

```python
results.append({
    "case": name, "passed": answer_ok and tool_ok,
    "answer": result["answer"], "trace": result["trace"],
    "memory_warning": result["memory_warning"],
})
```

`results.append(字典)` —— 往清单尾部**追加**一项。这是**列表最常用的方法**。

`"passed": answer_ok and tool_ok` —— 这一项**必须两项都合格**才算通过（`and` 短路与）。

**注意每一项都有 `case` 和 `passed` 两个键**——不管成功还是失败都有。这是刻意的：只有这两个键是**固定不变**的，后面的明细键可以因题而异（成功的给 `answer`/`trace`，失败的给 `error`）。这样函数末尾的计分代码才能统一处理。

**返回结果：**

```python
return result
```

`ask()` 把原始结果**原样交回去**，方便调用方接着用。

### 4.4 `except` 分支：一道题炸了不能让整个评测挂掉

```python
except Exception as error:
    results.append({"case": name, "passed": False, "error": type(error).__name__})
    return None
```

这是 `evaluate.py` 里**最重要的设计决策之一**。

**为什么不把异常往外抛？**

因为如果传出去，`main()` 那层的 `try/finally` 会关掉连接、整个脚本以失败告终——**后面十几道题一道都不跑了**。

而实际情况是：评测跑十几分钟、花掉几十次模型调用，**如果第 3 题因为偶发的网络抖动挂了，把剩下 9 题一起带走就太浪费了**。

所以这里的策略是：**把这道题记成"失败"，然后若无其事地继续下一道。**

这跟 `pytest` 里的一个测试失败不影响其他测试是同一个思路——**测试脚本要尽可能多地收集信息，而不是遇到第一个问题就戛然而止**。

**`type(error).__name__`** —— 和第 5 节 `app.py` 里见过的一模一样：取类型的**名字字符串**（比如 `"ValueError"`、`"TimeoutError"`）。

**注意这里只记了错误类型的名字，没记错误详情。** 为什么？

因为异常信息里**可能带着请求内容**（比如模型服务返回的原始报文，里面可能有你的提示词、甚至密钥片段）。报告文件是要落盘、可能要发给别人看的，**不该把这种东西写进去**。

这个考虑和 `main()` 最后那段（第 12 节）是同一个思路——**只暴露类型，不暴露内容**。

**两种情况都写报告**：成功也记、失败也记。所以这个函数**永远不会漏题**——报告逐条记录测试结果，这才能叫"体检报告"。

---

## 5. 第一批——三道"基本功"题

```python
prefix = "eval_" + run_id
ask("计算器", prefix + "_math", "请使用计算器计算 (12+8)*3。", ["60"], "calculator")
```

### 5.1 `prefix` —— 给这次评测的所有东西加个"姓"

`prefix = "eval_" + run_id` 得到类似 `eval_3f9a1c2b7d04` 的字符串。

后面所有会话 ID 都长这样：

```text
eval_3f9a1c2b7d04_math
eval_3f9a1c2b7d04_time
eval_3f9a1c2b7d04_rag
eval_3f9a1c2b7d04_memory
eval_3f9a1c2b7d04_new
eval_3f9a1c2b7d04_upload
```

**为什么要这么起名？** 两件事：

1. **每次评测的数据互不干扰**。跑十次评测就有十套独立会话，第 5 次的结果不会影响第 8 次。
2. **方便事后清理**。你一眼就能看出 `conversation_turns` 表里哪些行是评测留下的、属于哪一次。

> 这个技巧叫**命名空间前缀**，在"临时数据"和"真实数据"混在一张表时特别有用。

### 5.2 第一题：计算器

```python
ask("计算器", prefix + "_math", "请使用计算器计算 (12+8)*3。", ["60"], "calculator")
```

| 参数 | 值 |
| --- | --- |
| 名字 | `"计算器"` |
| 会话 | `eval_xxx_math` |
| 问题 | `请使用计算器计算 (12+8)*3。` |
| 必须包含 | `["60"]` |
| 期望工具 | `"calculator"` |

**这道题的检查点是双重的**：

- 答案里出现 `60` —— 说明**算对了**；
- 轨迹里出现 `calculator` —— 说明它**是调工具算的，不是自己心算编的**。

第二点比第一点更重要。**因为模型心算 `(12+8)*3` 也有可能蒙对**（这题太简单了），但它蒙对这件事**恰恰是危险的信号**——说明它在该用工具的时候没用工具。所以这里必须同时检查工具。

> 注意问题里的措辞 `"请使用计算器计算"`：这是**故意给的强提示**。评测要测的是"工具链路通不通"，不是"模型会不会自己想到用工具"，所以先把意图说清楚，减少不必要的随机性。

### 5.3 第二题：当前时间

```python
ask("当前时间", prefix + "_time", "请调用时间工具，并写出当前完整日期和时间。",
    [str(datetime.now().year)], "current_time")
```

**这道题的判定词是"动态生成"的**：

```python
[str(datetime.now().year)]
```

- `datetime.now()` 取此刻本机时间；
- `.year` 取**年份**部分，得到整数 `2026`；
- `str(...)` 把整数**转成字符串**，变成 `"2026"`；
- `[...]` 包成列表，因为 `ask` 要的是 `list[str]`。

**为什么必须动态算，不能把 `"2026"` 写死？**

因为**这道题明年就会失败**。写死的题在一次性的脚本里是"技术债"——今天能过，明年跑就是红的，而你还得花时间去查"到底哪里坏了"（其实只是年份变了）。

> **凡是"依赖当下时间"的断言，一律现场算。** 这是写测试的一条基本纪律。

**这道题只查工具（`current_time`），不查别的**：因为时间精确到分钟在变，没法用一个固定字符串去比。用年份是"足够松但又能证明它没瞎编"的折中——**如果它不调工具，模型凭训练记忆根本说不出 2026。**

### 5.4 第三题：知识问答（这道题的题名很有信息量）

```python
ask("无检索提示词的知识问答", prefix + "_rag",
    "星舟课程退款需同时满足哪些条件？保留 7 天和 20% 的数字写法并引用文件名。",
    ["7", "20%", "course.md#"], "knowledge_search")
```

题名叫 **"无检索提示词的知识问答"** —— 这是这道题最关键的地方。

**"无检索提示词"的意思是：问题里没有写"请查知识库"。**

回忆 `app.py` 那篇讲过的：每轮问答 `agent.py` 会**自动先检索一次知识库**（轨迹里标 `trigger: automatic`），不管用户说没说"请查询"。

所以这道题就是在验证这个能力：**用户正常问一句话，系统自己就去找了资料、还正确引用了来源。**

（反面的对照在第 9 节——那里也是"无检索提示词"，但查的是**临时上传**的文档。）

**检查三个词：**

| 判定词 | 在查什么 |
| --- | --- |
| `"7"` | 答案里有没有"7 天"这个关键数字 |
| `"20%"` | 有没有"学习比例不超过 20%" |
| `"course.md#"` | **有没有真的引用来源** |

**第三个词是最狠的**——`course.md#` 这个格式来自 `rag.py` 的 `context()` 内部拼接，它把每段证据标成 `[文件名#切片号]`。

模型被系统提示词要求"**用 `[文件名#切片号]` 引用**"，所以：

- 只要答案里出现了 `course.md#`，就说明**模型确实拿到了检索结果、并且按规矩标了出处**；
- 如果它靠自己瞎编，是编不出 `course.md#0` 这种精确格式的。

**这是在用"格式"反证"链路"** —— 一个很聪明的检查方式。它比"答案听起来对不对"可靠得多。

注意判定词写的是 `course.md#`（**带井号**），不是 `course.md` —— 加上井号才能确保它引用的是**具体某个切片**，而不是泛泛提了一嘴文件名。

---

## 6. 第二批：短期原文、真实摘要与摘要后的接续

先调用一次真实 `chat()`，让用户姓名与“星河练习”这个会话示例背景出现在旧问答中。姓名应提取到个人画像，示例背景保留为会话资料。

随后询问姓名，验证未摘要时原文和画像能正常使用。接下来只在本次独立评测应用中，临时复制配置，将 `context_token_budget=1`、`summary_keep_turns=1`，通过一次真实问答触发回答后摘要；`finally` 恢复原配置。

调低阈值是为了少量真实请求就能验证摘要链路，不必制造五万 token 的付费输入。直接 `save_turn()` 写占位记录不会运行摘要函数，不能用来证明摘要生成通过。

这组评测记录三项结果：

1. “真实回答后触发摘要”：真实 `chat()` 完成回答与摘要调用。
2. “摘要持久化与覆盖边界”：从 MySQL 重读，摘要非空且保留早期示例背景，只留下最新一轮未压缩原文，原文 ID 在覆盖位置之后。
3. “摘要后的历史背景接续”：恢复正常阈值后问早期项目代号，检查模型能通过摘要回答“星河练习”。

跨会话姓名、更正、遗忘及用户隔离仍由后面原有评测检查。摘要生成依赖真实模型，关键字判定可能误判，需要对照报告与数据库快照。

## 7. 抛开模型，直接查数据库复核

前面六道题都是"**看模型怎么答**"。这一段换了思路——**自己连一次数据库，验一遍数据**。

```python
recovered = ConversationMemory(app.settings)
other_user = ConversationMemory(replace(app.settings, user_id=prefix + "_other"))
```

### 7.1 为什么要**新建**一个 `ConversationMemory`？

`app.memory` 本来就是一个现成的 `ConversationMemory`，为什么不用它、非要新建一个？

**因为要验证"数据真的落盘了"，而不是"内存里恰好记着"。**

想象一种很糟的情况：`save_turn` 写了个 bug，只在内存里存了、压根没提交到数据库。那么用 `app.memory` 去查，**依然能查到**（因为它用的是同一个连接、同一份缓存）——测试就"通过"了，**但真相是这个系统的数据根本没存住**。

换个**全新的连接**去查，就等于**换一个人的身份去数据库里翻**：如果翻得到，说明数据**确实在库里**；翻不到，就说明之前的"保存成功"是假的。

> 这个模式叫"**独立复核**"：**不要用同一个对象去验证它自己的行为**。这也是为什么变量名叫 `recovered`（"重新取回来的"）——它在模拟"程序重启之后，数据还在不在"。

### 7.2 `replace(app.settings, user_id=...)` —— 造一个"另一个用户"

```python
other_user = ConversationMemory(replace(app.settings, user_id=prefix + "_other"))
```

这一行是第 2.2 节那个 `replace()` 的第一次实战。

它做的事：**拿 `app` 的配置当模板，只把 `user_id` 换成一个从没用过的新用户，其余（数据库地址、库名等）全部照抄。**

于是 `other_user` 就是一个**连着同一个 MySQL、但身份是"另一个人"**的记忆管理器。

**为什么不直接用 `USER_ID=local_user` 当那个"别人"？**

因为那样**会污染网页用户的画像**——万一下面的 `profile()` 或别的操作意外写了什么，"别人"就成了真实用户。所以这里另造一个 `eval_xxx_other`，**保证只碰评测自己的数据**。

这跟 `main()` 里换 `user_id="eval_" + run_id` 是同一个考虑（README 里写的"**不修改网页用户的画像**"）。

### 7.3 `try / finally` —— 保证连接一定被关掉

```python
try:
    ...
finally:
    recovered.close()
    other_user.close()
```

**`finally` 的含义是"不管上面发生了什么，这段一定执行"。**

| 情况 | `finally` 会执行吗 |
| --- | --- |
| `try` 里全部顺利跑完 | ✅ 会 |
| `try` 里抛了异常 | ✅ **照样会** |
| `try` 里有 `return` | ✅ **照样会** |

**为什么必须用 `finally` 而不是写在末尾？**

因为 `try` 块里有好几处可能抛异常（查数据库、读画像）。如果代码写在末尾，那么一旦中间出错，**执行会直接跳走，`close()` 永远轮不到**——两个数据库连接池就这么泄漏了。

`finally` 保证**无论走哪条路，连接都会还回去**。

这和 `app.py` 里的 `with` 语句是**同一类需求的不同写法**：

| 写法 | 适用 |
| --- | --- |
| `with ... as x:` | 资源生命周期**完全在**这个块里 |
| `try / finally` | 资源要在外面创建、**跨越很长一段代码**使用，中间还想自己控制异常 |

这里的 `recovered` 和 `other_user` 是在 `try` **外面**创建的（因为要用它们），所以只能用 `try/finally`。

### 7.4 重连后直接读取问答，验证会话隔离

```python
saved_turns = recovered.recent(memory_session)
restored = any("小明" in row["assistant_text"] for row in saved_turns)
isolated = not recovered.recent(prefix + "_other")
```

`recent()` 返回按旧到新排列的问答字典列表。`any(...)` 逐条看助手回答，至少一条包含“小明”，就说明新连接仍能读到已保存的回答。

另一个会话从未写入记录，读取结果应是 `[]`，所以 `not []` 得到 `True`。这里直接查数据库，不调用模型，也不经过关键词检索。

### 7.5 记一条结果

```python
results.append({"case": "MySQL 重连与会话隔离", "passed": restored and isolated})
```

**注意这一项只有 `case` 和 `passed` 两个键**——没有 `answer`、没有 `trace`。因为它**根本不是通过模型答出来的**，是直接查数据库得到的结论。

这也印证了第 4.3 节说的：`case` 和 `passed` 是**固定键**，其他明细因题而异。

### 7.6 画像持久化 + 用户隔离

```python
profile = recovered.profile()
name_keys = [key for key, fact in profile.items() if fact["value"] == "小明" and not fact["forgotten"]]
results.append({"case": "结构化记忆持久化与用户隔离", "passed": bool(name_keys)
                and not other_user.profile() and not other_user.recent(memory_session)
                and not other_user.list_sessions()})
```

**第一行**：用**独立连接**读一次画像——验证"姓名=小明"真的进了 `user_memories` 表。

**第二行：找出"哪一格存着小明"**

```python
name_keys = [key for key, fact in profile.items() if fact["value"] == "小明" and not fact["forgotten"]]
```

这是个**列表推导式 + 元组解包 + 过滤条件**，三个东西叠在一起（`app.py` 第 10.2 节见过字典版的同类写法）。拆开读：

| 片段 | 意思 |
| --- | --- |
| `for key, fact in profile.items()` | 挨个取画像的"字段名 + 事实内容"（元组解包） |
| `if fact["value"] == "小明"` | 只留下"值是小明"的那些 |
| `and not fact["forgotten"]` | **并且还没被标记为遗忘** |
| `[key for ...]` | 把这些字段名收集成列表 |

结果 `name_keys` 大致是 `["姓名"]`（具体叫什么取决于模型提取时起的字段名）。

**这个变量在后面的遗忘断言里还会再被用到**——这就是它为什么要在这里先算出来的原因（见第 8.3 节）。

**`not fact["forgotten"]`** 这个条件很必要：因为**遗忘操作会把 `value` 置为 `None`**，但这里如果只判断 `value == "小明"`，被遗忘后值就不是小明了，条件自然不成立。加上 `not fact["forgotten"]` 是**双保险**，语义也更明确："我要找的是活着的、叫小明的那个字段"。

**第三～五行：一个长长的 `and` 链条**

```python
"passed": bool(name_keys)
          and not other_user.profile()
          and not other_user.recent(memory_session)
          and not other_user.list_sessions()
```

Python 允许把一行表达式**折成多行**——只要它们都在**一对括号里面**（这里的括号是 `results.append({...})` 那对）。缩进随你，解释器不看缩进，只看括号没闭合。

> 这个技巧在你写长条件时很有用，不然一行能拖到 150 个字符。**注意：只有括号里能这么折；没有括号就必须用反斜杠 `\`，那就不推荐了。**

四个条件逐个看：

| 条件 | 在验证什么 |
| --- | --- |
| `bool(name_keys)` | **本用户的画像里确实有"小明"** |
| `not other_user.profile()` | **另一个用户的画像必须是空的**（`{}` 是假） |
| `not other_user.recent(memory_session)` | 另一个用户**读不到**这个会话的聊天 |
| `not other_user.list_sessions()` | 另一个用户的**会话列表也是空的** |

**这就是一道完整的"租户隔离"检查**：一个人能看到自己的东西，**并且绝对看不到另一个人的**。

后三个 `not` 里那个 `bool(name_keys)` 是**必需的**，不能省。为什么？见下一节。

### 7.7 为什么必须写 `bool(name_keys)`？（一个真实的坑）

如果没有 `bool(name_keys)` 这个条件，这一项**在出错的时候反而会"通过"**：

假设"提取姓名"这个功能**彻底坏了**——画像永远是空的。那么：

- `name_keys` = `[]`（空列表）；
- **`not other_user.profile()`** → 别的用户画像也是空的 → `True`；
- **`not other_user.recent(...)`** → 空 → `True`；
- **`not other_user.list_sessions()`** → 空 → `True`；
- 三个 `and` 全是 `True` → **`passed: True`** 😱

**"什么都没提取到"和"隔离得很好"在数据上长得一模一样**，光靠后三个条件根本分不出来。

所以必须在**开头**先钉一个**正向条件**：**"本用户确实有一条小明"**。有了它，功能坏掉时 `name_keys` 为空 → 整项就是 `False`，**正确地报错**。

> 这类问题叫"**空洞通过**"（vacuous truth）——**"所有元素都满足条件"在"一个元素都没有"时天然成立**。任何用 `all()` 或"所有 `not`"拼出来的检查，都要想一想"如果集合是空的会怎样"。这是写测试时最隐蔽的坑之一。

### 7.8 收尾

```python
finally:
    recovered.close()
    other_user.close()
```

两个连接池各自 `dispose()`（关闭并释放所有连接）。放在 `finally` 里，**保证异常时也会执行**。

> **注意这里没关 `app.memory`** —— 因为它**不是这里创建的**，得由创建它的 `main()` 负责。**谁创建谁销毁**，这是资源管理的一条铁律。

---

## 8. 第三批——更正与遗忘

这一组测的是"**用户改变主意时，系统跟不跟得上**"。也是三道题串成一条线。

### 8.1 换个会话，还记得我吗

```python
new_session = prefix + "_new"
ask("新会话共享个人记忆", new_session, "我叫什么名字？", ["小明"])
```

**在全新会话里问同一个问题。**

这个新会话 `eval_xxx_new` **从来没聊过天**，所以：

- `memory.context(new_session)` → **空的**，一条近期聊天都没有；
- `memory.profile()` → **还是有"姓名=小明"**。

**它应该答对。** 这就证明了那句最反直觉的设计：

> **聊天记录按会话分，个人画像按用户共享。**

当前会话早期背景通过摘要与未压缩原文继续使用；这道题切换到新会话，专门检查按用户共享的个人画像。新会话不继承旧会话摘要。

### 8.2 更正，并回到**旧会话**验证

```python
app.chat(new_session, "更正：我现在叫小舟，请更新我的姓名。")
ask("旧会话读取更正后的记忆", memory_session, "我现在叫什么名字？", ["小舟"])
```

**这两步的安排很妙，注意两个细节：**

**① 更正发生在 `new_session`，验证发生在 `memory_session`。**

**跨会话**——这正是要证明的东西：**更正作用于"这个用户的画像"，所以所有会话立刻都能看到新值。**

而 `memory_session` 里那些老聊天记录里**明明写着"小明"**（准备动作里种下的）。如果模型不小心看了旧历史，就会答"小明"——**这道题实际上在验证"模型有没有正确地让新画像压过旧历史"**。

这就是 `SYSTEM_PROMPT` 里那两句规则在起作用：

> `本轮用户明确更正优先；最新有效画像优先于旧会话历史中的冲突事实。`

**② 第一句用 `app.chat(...)`，第二句用 `ask(...)`。**

和 6.1 节一个道理：**第一句是"准备动作"（制造更正），第二句才是"检查动作"**。

如果第一句也用 `ask(...)` 并给了判定词，那它就是在同时测"模型接不接受更正指令"——但这不是这组题的目标。**一次只测一件事**，职责才清楚。

**注意问答的用词**：问题是 `我现在叫什么名字？`，判定词是 `["小舟"]`。

如果模型把两个名字都说出来（"你原来叫小明，现在叫小舟"），这道题**照样能过**——因为 `"小舟" in 回答` 成立。这是**关键词断言偏宽松**的一面（对比 4.3 节说的偏严格那一面）。**两种偏差都存在，这也是为什么报告里一定要留 `answer` 原文。**

### 8.3 遗忘，并且**验证标记落在正确的字段上**

```python
app.chat(new_session, "请忘记我的姓名，别再把它作为当前个人资料。")
profile = app.memory.profile()
results.append({"case": "结构化记忆弃用标记", "passed": bool(name_keys)
                and all(profile.get(key, {}).get("forgotten") for key in name_keys)})
```

**遗忘这一步**：让系统忘记姓名。注意措辞是 `"别再把它作为当前个人资料"`——这是**刻意说清楚的**，因为"遗忘"在这套系统里**不是删除**，只是"不再当现状用"（原始聊天和文档都留着）。

**随后重新读画像**（这次用的是 `app.memory` 而不是 `recovered`，因为 `recovered` 已经在 7.8 节关掉了）。

**最后这个断言是整个文件里最精巧的一处。**

```python
all(profile.get(key, {}).get("forgotten") for key in name_keys)
```

**核心思路**：不是随便找个人问"姓名忘了没"，而是 **`name_keys` —— 前面算出来的那个"当时存着小明的字段名"**，专门去检查**那一格**的 `forgotten` 标记。

**为什么要绕这一下？**

因为**遗忘最容易出的 bug 是"忘错了字段"**。假设画像里有两条：`姓名=小明` 和 `学习目标=掌握 Python`。如果程序在遗忘时**不小心把"学习目标"标记了**，那么：

- 如果你只检查"画像里还有没有 `forgotten=True` 的字段" → **有**（学习目标被误标了）→ 测试**错误地通过**；
- 但用 `name_keys` 精确定位到"姓名"那一格去查 → 发现"姓名"根本没被标记 → **正确地报错**。

**这就是"用具体字段名而不是模糊条件"的价值**：能把"忘错对象"这种 bug 抓出来。

**再看 `profile.get(key, {}).get("forgotten")` —— 为什么套两层 `.get()`？**

```python
profile.get(key, {})        # 第一步：取"姓名"这一格
       .get("forgotten")    # 第二步：取这一格里的"forgotten"键
```

- **外层 `profile.get(key, {})`**：用 `.get()` 而不是 `profile[key]`，是为了**万一 key 不在画像里也不报 `KeyError`**。取不到时给一个空字典 `{}` 兜着。
- **内层 `.get("forgotten")`**：取不到就返回 `None`。而 `None` 在布尔判断里算"假"——**正好等于"没被标记为遗忘"**，语义上刚好正确。

**这是"层层兜底"的写法**：任何一层取不到，都不会崩，只会得到"假"。

> 反过来说：如果你写 `profile[key]["forgotten"]`，只要 `key` 不存在就立刻 `KeyError`，整个评测挂掉。**读"可能不存在"的数据，用 `.get()`；读"保证存在"的数据，用 `[]`。**这个判断在 `app.py` 那篇（第 10.4 节）也提到过。

**仍然是 `bool(name_keys) and ...` 起头** —— 和第 7.6 节同一个防"空洞通过"的考虑。

**注意数字出现顺序**：`name_keys` 在前面算出，用在遗忘断言。中间隔着十几行代码和一次更正。**这里之所以成立，是因为"更正是改值、不改字段名"**——`姓名` 这个 key 从始至终没变，变的只是它的 `value`。

> 这是个**隐含假设**：它依赖"模型在更正时复用了同一个字段名"。系统提示词里确实要求了"同一含义复用同一 key"（`structured_memory.py` 的 `EXTRACTION_PROMPT`），但**如果模型没遵守，这道题会失败**。这属于"评测可能受模型波动影响"的一个真实例子——报告里每一项都能看到 `answer` 原文，就是为了这种时候人工复核用。

---

## 9. 第四批——上传一份临时文档再考它

```python
name = "evaluation_notes.txt"
app.knowledge.upload([(name, "极光计划接头口令是蓝色松果。".encode("utf-8"))])
ask("无检索提示词的上传问答", prefix + "_upload",
    "极光计划接头口令是什么？请引用文件名。",
    ["蓝色松果", "evaluation_notes.txt#"], "knowledge_search")
```

### 9.1 `.encode("utf-8")` —— 把文字变成字节

```python
"极光计划接头口令是蓝色松果。".encode("utf-8")
```

**这是这一行最需要理解的语法。**

字符串（`str`）在 Python 里是"**一串字符**"，而文件内容在网络和磁盘上是"**一串字节**"（`bytes`）。两者之间要**编码（encode）**来转换：

```python
text = "极光计划接头口令是蓝色松果。"        # str，给人看的
data = text.encode("utf-8")                  # bytes，给机器存的
```

反过来是 `.decode("utf-8")`（`documents.py` 里读 txt 文件用的就是它）。

**为什么必须是 utf-8？** 因为中文字符在别的编码（比如 GBK）下会变成完全不同的字节。`documents.py` 里就专门检查过这一点——如果上传的 txt 不是 UTF-8，会直接抛错 `"必须保存为 UTF-8 编码"`。这里既然是我们自己造的字节，就用最通用的 UTF-8。

**为什么需要字节，而不是直接传字符串？**

因为 `upload()` 的接口是 `list[tuple[str, bytes]]`（`app.py` 那篇第 9.7 节讲过）——它模拟的是"**用户从硬盘选了一个文件**"，而文件本来就是字节。**评测要模拟真实的上传路径，就得给一样类型的东西。**

### 9.2 `[(name, "...".encode("utf-8"))]` —— 为什么是个"列表套元组"

拆开看这个数据结构：

| 层 | 是什么 | 为什么 |
| --- | --- | --- |
| 最外层 `[...]` | **列表** | 因为上传接口支持**一次传多个文件** |
| 里面 `(name, data)` | **元组** | 一个文件就是"**文件名 + 内容**"这一对 |

所以整体读作："**一份文件，叫 `evaluation_notes.txt`，内容是这样的字节。**"

`name` 变量先赋值再用，是为了让**文件名只写一次**（下面上传题的判定词里还要再提一次这个名字）。

### 9.3 这道题的判定词有讲究

```python
["蓝色松果", "evaluation_notes.txt#"], "knowledge_search"
```

| 判定词 | 在查什么 |
| --- | --- |
| `"蓝色松果"` | 答案里有没有那串**杜撰的接头口令** |
| `"evaluation_notes.txt#"` | 有没有**正确引用这份刚上传的文件** |

**为什么用"极光计划 / 蓝色松果"这种故意编造的内容？**

因为这两个词**在模型的知识里根本不存在，在网上也搜不到**。所以：

> **只要答案里出现了"蓝色松果"，就 100% 说明它读的是我们刚上传的那份文档。**

这是个很漂亮的**反证设计**。如果用的是"星舟课程退款"这种预置文档里的内容，你没法排除"模型本来就听说过"的可能性。**用不存在的词，才能干净地把"检索"和"记忆"分开。**

### 9.4 一个需要知道的副作用

```python
app.knowledge.upload([...])
```

这一行会**真的往磁盘上写一个文件**：

```text
data/evaluation/<run_id>/uploads/evaluation_notes.txt
```

因为 `main()` 里把 `data_dir` 换成了这次评测专属的目录（第 11.3 节），所以**它不会污染网页用户的上传目录**（`data/uploads/`）。每次评测都会在自己的小文件夹里留一份，互不干扰。

> 这也解释了为什么 `evaluate.py` **跑完不能立刻删掉 `data/evaluation/`**——报告和上传的测试文件都在里面。想清理的话，README 里说过：删整个 `data/evaluation/` 目录即可。

---

## 10. 算总分，交报告

```python
passed = sum(row["passed"] for row in results)
return {"passed": passed, "total": len(results), "results": results}
```

### 10.1 `sum(row["passed"] for row in results)`

又是"生成器表达式 + 内置函数"的组合（和第 4.3 节的 `all(...)` 同一家族）。

`results` 里每一项都有个 `passed` 键，值是 `True` 或 `False`。`sum(...)` 把它们加起来。

**为什么 `True`/`False` 能相加？**

这是 Python 一个很有用的历史设计：**`bool` 是 `int` 的子类**。

```python
True + True + False + True      # → 3
isinstance(True, int)           # → True
```

在数值运算里，`True` 当 `1` 用、`False` 当 `0` 用。所以 `sum` 直接就把"**有几项通过**"数出来了，**不需要写循环、不需要 if 计数**。

> 顺便一提：这个设计也有反面——`True + 1` 等于 `2` 而不是报错。所以在 `tools.py` 的 `calculator` 里，作者**特意用 `type(node.value) not in (int, float)` 而不是 `isinstance`**，就是为了把 `True` 挡在外面（`app.py` 那篇提过这个细节）。**同一个语言特性，在一处是便利、在另一处是坑。**

### 10.2 返回的字典就是报告的内容

```python
{"passed": passed, "total": len(results), "results": results}
```

| 键 | 值 | 说明 |
| --- | --- | --- |
| `passed` | 通过几项 | 比如 `13` |
| `total` | 总共几项 | `len(results)`，关闭反思时 **13**，开启时 **15** |
| `results` | 逐题明细 | 列表，每项含 `case` / `passed` / 明细 |

`len(results)` 这个写法很稳：**不用把"12"写死**。以后加题、减题，`total` 自动跟着变，不会对不上。

> 这是"**不要手写能被算出来的常量**"的一个小例子。硬写 `"total": 13` 在今天没错，但改代码的人一定会忘。

---

## 11. `main()` —— 搭台、唱戏、收拾、写报告

```python
def main() -> None:
    """建立独立评测环境并运行用例，报告写入 data/evaluation/<运行ID>/。"""
```

### 11.1 先发一个编号

```python
run_id = uuid4().hex[:12]
```

- `uuid4()` 生成一个 32 位十六进制随机串；
- `.hex` 取它的十六进制形式（**字符串**）；
- `[:12]` **切片** —— 只取**前 12 位**。

**为什么要截短？** 因为完整 32 位太长，而会话 ID 长这样：

```text
eval_3f9a1c2b7d04_memory
```

12 位随机 + 前缀，既**足够唯一**（16¹² 种可能），又**看着不累**——目录名、会话名、报告路径里都要用它，短一点到处都清爽。

### 11.2 算出报告应该放哪

```python
folder = ROOT / "data" / "evaluation" / run_id
```

**这行用的是 `pathlib` 的"斜杠拼路径"写法**——`Path` 对象的 `/` 运算符被重载成了"拼接"（不是除法）。

`ROOT` 是 `config.py` 里算出来的项目根目录，所以这一行的结果是：

```text
D:\study\AI\ai-agent-demo-study\data\evaluation\3f9a1c2b7d04
```

**为什么用 `ROOT / ...` 而不是写死 `"data/evaluation/..."`？**

因为**相对路径是跟着"当前工作目录"走的**。如果你在别的目录下敲命令，写死的相对路径就会把报告扔到别的地方去。用 `ROOT` 拼出来的**绝对路径**，不管在哪儿执行都指向同一个地方。

> 这是 `pathlib` 相对字符串拼路径的核心优势——**`Path` 会正确处理不同系统的分隔符，也能 `resolve()` 出真正的绝对位置。**

### 11.3 造一份"评测专用的配置"

```python
settings = replace(load_settings(), data_dir=folder, user_id="eval_" + run_id)
```

**这一行是整个 `evaluate.py` 隔离机制的总开关。** 它一次换掉两个字段：

| 字段 | 原本 | 换成 | 效果 |
| --- | --- | --- | --- |
| `user_id` | `local_user` | `eval_3f9a1c2b7d04` | 换一个**评测专用用户** |
| `data_dir` | `data/` | `data/evaluation/3f9a1c2b7d04/` | 换一个**评测专用数据目录** |

**换 `user_id` 解决了什么？**

`agent_demo` 里所有数据读写都认这个 `user_id`。换成 `eval_xxx` 之后：

- 评测写进 `user_memories` 的画像**属于 `eval_xxx`**，**碰不到 `local_user` 的画像**；
- 聊天记录也一样，网页用户打开界面**看不到任何评测留下的对话**。

**这就是 README 里那句"不修改网页用户的画像"的实现方式。** 而且注释写得很直白：

```python
# 独立用户和目录：评测聊天、画像、上传和向量库都不碰网页用户。
```

**换 `data_dir` 解决了什么？**

因为 `data_dir` 一旦改变，跟着变的是一串东西。回顾 `Settings` 里依赖它的字段：

| 依赖 | 原位置 | 评测时位置 |
| --- | --- | --- |
| 上传文件目录 | `data/uploads/` | `data/evaluation/<id>/uploads/` |
| Chroma 向量库 | `data/chroma/` | `data/evaluation/<id>/chroma/` |

于是评测用的是**一个全新的空向量库**——它会从零开始把 `docs/knowledge/` 里的文档向量化一遍（**这一次要真花钱**，因为要调 Embedding 接口）。

**为什么不复用主库的索引省点钱？**

因为**评测应该是"从零开始"的**。用旧索引会让结果不可复现——上次跑留下的脏数据、旧版本的切片可能混在里面。**每次评测都在干净的盒子里跑，结果才可信。**

而且这样也不会污染主库的索引。

> **注意 `replace()` 的两处用法在这一点上是呼应的**：`main()` 里用它换 `user_id` + `data_dir`（造环境），第 7.2 节里用它换 `user_id`（造"另一个人"）。同一个工具，两种隔离需求。

### 11.4 建应用、跑评测、保证关连接

```python
app = AgentApp(settings)
try:
    report = run_evaluation(app, run_id)
finally:
    app.memory.close()
```

**为什么 `app = AgentApp(settings)` 放在 `try` 外面？**

因为**如果建 `app` 这一步就失败了，就没有 `app` 可以关**。放进 `try` 里，`finally` 去调 `app.memory.close()` 会报 `NameError`（`app` 这个名字压根没被赋值）——**用一个错误掩盖另一个错误**。

放外面就干净了：**建好了才进 try，进了 try 就保证会关。**

**`finally: app.memory.close()`** —— 只关 `memory`（数据库连接池）。

**为什么只关这一个？** 因为 `AgentApp` 持有的三样东西里，只有 `memory` 是"需要显式释放的连接池"（`llm` 和 `knowledge` 都是 HTTP 客户端，由各自的库自己管理）。所以 `close()` 关到点子上就够了。

**注意 `report` 是在 `try` 里赋值的**，但**在 `finally` 之后还要用**（后面写文件、打印时）。这在 Python 里是合法的——`with`/`try` 块**不会创建新的作用域**，块里赋的变量在外面照样能用。

> 这一点和很多语言不同（也和 `if` / `for` 一样）：**Python 只有函数、类、模块会形成作用域，`if`、`for`、`try` 都不形成。**所以 `report` 出了 `try` 还在。

### 11.5 写报告文件

```python
folder.mkdir(parents=True, exist_ok=True)
path = folder / "report.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
```

**`folder.mkdir(parents=True, exist_ok=True)` 的两个参数：**

| 参数 | 含义 | 不写会怎样 |
| --- | --- | --- |
| `parents=True` | **父目录不存在也一起建**（相当于 `mkdir -p`） | 只建最后一级，父级不存在就报错 |
| `exist_ok=True` | **目录已存在也不报错** | 已存在就抛 `FileExistsError` |

其实这里 `folder` 大概率**已经存在了**——因为 `AgentApp` 初始化 Chroma 时已经在它下面建了 `chroma/` 目录。所以这一行是**防御性的**：不管前面发生了什么，先确保目录在，**后面写文件就不会失败**。

> 这是"**幂等**"的思路：这行代码执行一次和执行十次效果一样，可以放心重复调用。

**`path.write_text(...)` 三个要点：**

① **`write_text` 是 `pathlib` 的方法**，一行搞定"打开文件 → 写入 → 关闭"（用传统写法要三行 `open`/`write`/`close`，还得自己保证关闭）。

② **`json.dumps(report, ...)` 把字典转成 JSON 文本**。两个参数都很重要：

| 参数 | 作用 | 不写的后果 |
| --- | --- | --- |
| `ensure_ascii=False` | 中文**原样输出**，不转成 `\uXXXX` 转义 | 报告里全是 `\u6211\u53eb...`，**人根本读不了** |
| `indent=2` | **缩进 2 空格**，排版成多行 | 挤成一整行，**没法看** |

**`ensure_ascii=False` 是处理中文时最容易忘的一个参数。** 它的默认值是 `True`——意思是"为了绝对安全，把所有非 ASCII 字符都转成转义序列"。功能上没问题（转回来一样），但**人眼看着就是一堆乱码**。

③ **`encoding="utf-8"`** —— 明确指定编码。这很重要：**Windows 上默认编码不一定是 UTF-8**（可能是 GBK），显式写出来才能保证中文在各系统上都正常。

> **一条通用纪律**：Python 里读写**文本文件**时，**永远显式写 `encoding="utf-8"`**。不写就是在赌运行环境的默认值，而这个赌注在 Windows 上经常输。

### 11.6 在终端里说结果

```python
print(f"评测结果：{report['passed']}/{report['total']} 通过")
print(f"报告：{path}")
```

两句 `print` 输出到**终端**（不是网页、不是文件）。跑的时候你会看到：

```text
评测结果：11/11 通过
报告：D:\study\AI\ai-agent-demo-study\data\evaluation\3f9a1c2b7d04\report.json
```

**注意 f-string 里的引号嵌套**：外层是双引号 `f"..."`，里层取键用的是单引号 `report['passed']`。

**为什么不能都写双引号？** 因为 `f"评测结果：{report["passed"]}..."` 会让 Python 分不清**哪对引号是结束符**——内层的 `"` 会被当成字符串的结尾。所以**里层必须换一种引号**。

在 **Python 3.12 之前**这是硬规矩。**3.12 起放宽了**，允许在同一层 f-string 里复用引号（PEP 701），但**写成单双引号混用依然是更稳、更通用的写法**——毕竟你的 `.venv` 里可能是 3.11。

> 顺带一说：这个项目要求 Python **3.11～3.13**，所以代码里保持单双引号混用是必要的习惯。

### 11.7 用"退出码"报告成败

```python
raise SystemExit(0 if report["passed"] == report["total"] else 1)
```

**这是全文件的收尾，也是它作为一个"合格评测脚本"的标志。**

**先看 `0 if ... else 1`** —— 又是一个三元表达式（`app.py` 那篇第 5 节见过）：全过 → `0`，否则 → `1`。

**再看 `SystemExit`** —— 它是 Python 内置的一个特殊的"异常"。

为什么要用"抛出异常"的方式结束程序？因为在 Python 里，**`raise SystemExit(码)` 和 `sys.exit(码)` 是同一个效果**：

- **立即终止程序**；
- **把这个数字作为进程的"退出码"**交给操作系统。

**退出码的约定**（POSIX 标准，所有系统都认）：

| 退出码 | 含义 |
| --- | --- |
| `0` | **成功**（一切正常） |
| 非 `0` | **失败** |

**为什么非要在意这个数字？**

因为**它是给机器看的，不是给人看的**。人看终端里那行"11/11 通过"，但**自动化系统看的是退出码**：

```bash
python evaluate.py
if [ $? -ne 0 ]; then
    echo "评测没过，中止发布"
fi
```

在 CI/CD 流水线里，**"退出码非 0"就是"这一步失败"**，会自动中断后续步骤、给提交打红叉、发通知。

所以这行的真实含义是：**"评测通过就放行，没通过就让整个自动化流程停下来。"** 这是把"体检"接进"自动化门禁"的关键一步。

README 里那句"全通过退出码为 0，否则为 1"说的就是它。

---

## 12. 脚本入口与最后一道防线

```python
if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except Exception as error:
        logger.exception("评测运行失败")
        # 自己抛的 ValueError 给人看原话，其他异常只报类型，避免泄露请求内容。
        detail = str(error) if isinstance(error, ValueError) else "请检查模型服务、MySQL 和网络"
        raise SystemExit(f"评测失败（{type(error).__name__}）：{detail}") from None
```

### 12.1 `if __name__ == "__main__":` —— 命令行脚本的入口

这行是**全 Python 最著名的一行**。`app.py` 末尾也有它，但两者的启动方式不同：`evaluate.py` 直接由 Python 执行，`app.py` 则由 `streamlit run` 执行。

**`__name__` 是什么？**

每个 Python 文件（模块）都有一个内置的名字 `__name__`，取值规则是：

| 这个文件怎么被使用 | `__name__` 的值 |
| --- | --- |
| **直接运行**（`python evaluate.py`） | `"__main__"` |
| **被导入**（`import evaluate`） | `"evaluate"`（模块名） |

所以 `if __name__ == "__main__":` 的意思是：

> **"只有当这个文件是被直接运行的，才执行下面的代码；如果是被别人 import 的，就跳过。"**

**为什么 `evaluate.py` 需要它？**

两个理由：

1. **防止导入时误跑评测**。如果哪天有人想复用 `run_evaluation()` 这个函数（比如写一个自己的评测脚本 `from evaluate import run_evaluation`），**没有这个判断的话，`import` 的瞬间就会真的发起一整轮评测**——花钱、写数据库、覆盖报告。灾难。
2. **它本来就是个命令行脚本**，需要一个明确的入口。这个判断就是最标准的入口写法。

**那 `app.py` 为什么也写这一行？**

因为用 `streamlit run app.py` 时，脚本同样以 `__main__` 执行，`main()` 会被调用；加这一行只是避免被其他代码 `import app` 时误画页面。Streamlit 不会因此变成普通 Python 程序。

> 反过来说，这正是**"为什么不能 `python app.py` 直接跑"**的根本原因：那样 `__name__` 确实是 `"__main__"`，但**没有 Streamlit 服务器在监听**，`st.title()` 这些调用无处可画，只会得到一堆警告和一片空白。

### 12.2 最外面那层 `try/except`

```python
try:
    main()
except Exception as error:
```

**这里是整个程序的最后一道防线。**

`main()` 里虽然已经有 `try/finally` 了，但那个 `finally` **只负责关连接，不负责处理异常**——异常会继续往上冒。冒到 `if __name__ == "__main__":` 这一层如果没人接，Python 就会打印一大段原始 traceback（里面可能有敏感信息）。

所以这里兜住，**把异常转换成一句干净的、给用户看的话**。

### 12.3 那个三元表达式（第三次出现了）

```python
detail = str(error) if isinstance(error, ValueError) else "请检查模型服务、MySQL 和网络"
```

**这行和 `app.py` 里的 `show_error()` 一模一样**（第 5 节），连判断逻辑都一致：

| 异常类型 | 显示什么 | 为什么 |
| --- | --- | --- |
| `ValueError` | **原话** | 我们自己主动 `raise` 的，文字本来就是给人看的 |
| 其他异常 | `"请检查模型服务、MySQL 和网络"` | 原始信息里**可能有请求内容甚至密钥** |

**注意这里的兜底提示比 `app.py` 更具体**——它直接点名了三个最可能的原因：模型服务、MySQL、网络。因为跑评测时这三样任一不通都会失败，**直接告诉用户去检查哪里**，比一句"出错了"有用得多。

注释也写得很明确：

```python
# 自己抛的 ValueError 给人看原话，其他异常只报类型，避免泄露请求内容。
```

### 12.4 `raise SystemExit(f"...") from None` —— 最后两个知识点

**① `SystemExit` 的参数这次是个字符串**

```python
raise SystemExit(f"评测失败（{type(error).__name__}）：{detail}")
```

`SystemExit` 的参数**可以是数字（当退出码），也可以是字符串（当消息）**：

| 参数 | 效果 |
| --- | --- |
| `SystemExit(0)` | 退出码 0，**不打印任何东西** |
| `SystemExit(1)` | 退出码 1，**不打印任何东西** |
| `SystemExit("文字")` | **把文字打印到标准错误**，退出码为 1 |

所以这里给字符串，等于"**打印一句人话 + 以失败退出**"，一举两得。跑失败时终端会看到：

```text
评测失败（AuthenticationError）：请检查模型服务、MySQL 和网络
```

**注意**：`type(error).__name__` 只给了**类型名**（`AuthenticationError`），没有给原始信息——正好呼应注释里"只打印错误类型"。

**② `from None` 在做什么？**

这是本文件最后一处、也是最容易被忽略的一个语法。

**先看没有 `from None` 会怎样。** Python 在"处理异常的过程中又抛出新异常"时，会自动把两者串起来打印：

```text
Traceback (most recent call last):
  File "evaluate.py", line 172, in <module>
    main()
  File "evaluate.py", line 153, in main
    app = AgentApp(settings)
  ...
AuthenticationError: invalid api key: sk-xxxxxxxx       ← 原始异常被打印出来

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  ...
SystemExit: 评测失败（AuthenticationError）：请检查模型服务、MySQL 和网络
```

**看到问题了吗**：中间那句 `AuthenticationError: invalid api key: sk-xxxxxxxx` ——**原始异常被完整打印了，密钥就在终端里。**

`from None` 的作用就是**切断这条链**。它的技术含义是把新异常的 `__suppress_context__` 设为 `True`，效果是：

> **Python 不再把"原始异常"作为上下文一起打印出来。**

于是终端只剩下干净的一句：

```text
评测失败（AuthenticationError）：请检查模型服务、MySQL 和网络
```

**这就是注释里"密钥不会输出"的实现方式**——不是靠删除字符串，而是**靠不打印那个异常对象**。

> **这是个非常值得学的安全习惯**：处理来自外部服务的异常时，**不要让原始异常"漏"出去**。因为你不确定它的消息里到底装了什么——可能是密钥、可能是请求体、可能是服务端返回的内部细节。
>
> **`raise 新异常 from None` 是切断这条泄漏路径的标准手法。**

**注意一个边界**：`from None` 只切断"原始异常的上下文"，**新异常自己的 traceback 还是会打印**（因为它自己就是那次"崩溃"）。真正想完全不打印任何 traceback，需要更复杂的顶层处理——但对于这个教学项目，"不泄漏原始异常"就已经达到目的了。

---

## 13. 完整走一遍真实评测

```text
读取配置并复制独立评测用户、data_dir
  → 初始化实际 MySQL、Chroma 与模型客户端
  → 检查计算器、时间、自动知识检索
  → 建立姓名与练习背景，检查未摘要时的问答
  → 临时调低阈值，通过真实 chat 触发摘要并恢复配置
  → 从数据库核对摘要正文与覆盖位置，再问早期背景
  → 验证重连、用户与会话隔离、跨会话更正和遗忘
  → 上传虚构文档并提问
  → 开启反思时追加错稿与对稿两项验证
  → 写 report.json 并以退出码报告结果
```

默认 13 项、开启 Reflection 共 15 项。报告总数仍由 `len(results)` 计算；文中的其他旧示例分数只演示报告格式。准备阶段的直接模型调用、上传或数据库复核失败可能中断整场；`ask()` 的单题异常捕获并不覆盖所有准备动作。

## 14. 常见疑问

**Q1：跑一次要花多少钱？**

主要花在两处：① 初始化时把预置文档向量化一次（只在新建向量库时发生）；② 十几道题每道至少一次模型调用，其中"提取记忆"那一步还会**额外再调一次**。**所以十几道题大约等于二三十次模型请求的量级。**

值得注意的是：**`evaluate.py` 用的是全新的向量目录，所以每次跑都要重新向量化一遍预置文档。** 想省钱的话，别闲着没事反复跑。

**Q2：为什么我改了 `docs/knowledge/` 里的文档，评测结果没变？**

不会的——评测的 `data_dir` 是独立的，每次跑都是**从零建索引**，所以它**总是用最新的文档**。反过来说，网页那边的知识库**不会**因为跑评测而变化（它用的是 `data/chroma/`）。

**Q3：评测失败了，怎么知道是哪道题挂了？**

打开报告：

```text
data/evaluation/<run_id>/report.json
```

找 `passed` 为 `false` 的那些项。**注意不要只看 `passed`**，要看三个东西：

| 看什么 | 能发现什么 |
| --- | --- |
| `answer` | 模型实际答了什么（可能是**同义表达被判错**了） |
| `trace` | 它到底调了哪些工具、参数和结果是什么（**最有信息量**） |
| `memory_warning` | 记忆提取环节有没有出问题 |

如果 `answer` 里分明有正确答案、只是措辞不一样，那就是**关键词断言的误判**，不是真的 bug。

**Q4：评测会不会污染我平时的聊天和画像？**

**不会。** 这是这个文件设计上最讲究的地方——它换了 `user_id`（数据隔离）和 `data_dir`（文件隔离），从头到尾只碰 `eval_<run_id>` 这个名字下的东西。

**唯一的例外是"上传的那份文件"**——但那也在 `data/evaluation/<run_id>/uploads/` 里，不在你的 `data/uploads/` 下。

**Q5：能不能加自己的测试题？**

能，而且很简单——在 `run_evaluation` 里加一行：

```python
ask("我的新题", prefix + "_mine", "你的问题？", ["期望包含的词"], "expected_tool")
```

`total` 会自动跟着变（因为它用的是 `len(results)`）。**唯一要留意的是"准备动作"和"检查动作"要分开**——需要先布置状态的，用 `app.chat(...)` 或 `app.memory.save_turn(...)`，不要用 `ask(...)`。

**Q6：为什么有的题检查了 `expected_tool`，有的没检查？**

因为**能可靠预判的工具才值得检查**。

- "请使用计算器" → 一定能预判它该调 `calculator`；
- "我叫什么名字" → 依据本轮原话、当前有效画像、会话摘要和未压缩历史回答；缺少依据时说明不知道，不能声称已经查询过未提供的旧聊天。新会话只能使用共享画像，不继承其他会话的摘要。

**叫不准的时候就不写** `expected_tool`，只查答案内容——**避免把"模型的合理选择"误判成失败**。

**Q7：为什么不能直接 `python -c "import evaluate; evaluate.main()"`？**

可以，能跑。但**用 `if __name__ == "__main__"` 的理由**在于：别人 `import evaluate` 只想复用 `run_evaluation` 函数时，**不会被强制跑一遍评测**。这是这个判断存在的全部意义。

---

## 15. 语法速查表

对照 `app.py` 那篇的第 13 节，这里列的是**本文出现但那边没有、或者用法不同**的写法：

| 写法 | 白话 |
| --- | --- |
| `replace(对象, 字段=新值)` | `dataclasses` 的函数：**照着原对象复制一份**，改掉指定字段。因为 `Settings` 是 `frozen=True` 的，只能这么改 |
| `@dataclass(frozen=True)` | 冻结的数据类：**创建后字段不许改**，要变就 `replace()` 造一份新的 |
| `Any` | 类型标注里的"万能牌"：**什么类型都行** |
| `str \| None` | 类型是"字符串 **或** 无" |
| `参数: T = None` | **默认值**：调用时不传就取 `None` |
| `x is None` | **判断"是不是空"的标准写法**（不要用 `== None`） |
| `all(gen)` / `any(gen)` | 是不是**全部**为真 / **任意一个**为真 |
| `(x for x in data)` | **生成器表达式**：惰性流水线，配 `all` / `sum` 用最省 |
| `[f(x) for x in data]` | **列表推导式**：立刻造出一个新列表 |
| `for key, fact in d.items()` | **元组解包**：一次拿到键和值 |
| `for _ in range(n)` | `_` 是**约定俗成的占位符**："这个变量我不用" |
| `range(n)` | 产生 `0` 到 `n-1`，共 `n` 个 |
| `[start:stop]` | **切片**，`hex[:12]` 取前 12 位 |
| `bool(x)` | 把任意值转成 `True` / `False`（空列表、`0`、`""` → `False`） |
| `len(x)` | 数一数有几个 |
| `sum(gen)` | 求和。**`True` 当 `1`、`False` 当 `0`**，所以能直接数"通过了几项" |
| `d.get(k, 默认值)` | 取字典的键，取不到就用默认值（**不报错**） |
| `d.get(a, {}).get(b)` | **层层兜底**：任何一层取不到都只得到 `None`，不会崩 |
| `dict1 + "文字"` → `str + str` | 字符串拼接 |
| `"汉字".encode("utf-8")` | **编码**：字符串 → 字节（反向是 `.decode`） |
| `Path / "子目录"` | `pathlib` 的**斜杠拼路径**，不是除法 |
| `.mkdir(parents=True, exist_ok=True)` | 建目录：**父级一起建**、**已存在也不报错**（幂等） |
| `.write_text(文本, encoding=...)` | 一行完成"打开→写→关"，纯文本文件 |
| `json.dumps(对象)` | Python 对象 → JSON **字符串**（`indent` 缩进、`ensure_ascii=False` 保中文） |
| `try / finally` | 不管出没出错，**`finally` 一定执行**（用来保证资源被释放） |
| `print(...)` | 输出到**终端**（不是网页、不是文件） |
| `raise SystemExit(码)` | 立刻结束程序，**数字当退出码**（0 = 成功） |
| `raise SystemExit("文字")` | 结束程序：**打印文字 + 退出码 1** |
| `raise X from None` | **切断"原始异常"的打印链**，避免它带着密钥/请求内容漏出来 |
| `if __name__ == "__main__":` | **只有直接运行才执行**；被 `import` 时跳过 |
| 括号内折行 | 表达式太长时，**只要在括号里面就可以换行**，缩进随意 |
| `f"...{d['k']}..."` | f-string 里**单双引号混用**（3.12 之前必须这样） |

**"没有出现"的东西，同样值得留意：**

| 没有出现 | 为什么 |
| --- | --- |
| 任何第三方库的导入 | 这是个**命令行脚本**，用不到 Streamlit；模型/数据库都由 `agent_demo` 包封装了 |
| `st.xxx()` | 不在网页里跑，终端输出用 `print` |
| `pytest` 相关的东西 | 这是**独立评测**，不是单元测试（见第 0 节） |
| 硬编码的 `total = 11` | 用 `len(results)` 算，**加减题自动同步** |
| 硬编码的年份 `"2026"` | 用 `datetime.now().year` 现场算，**明年也不会失效** |
| `settings.user_id = ...` | `Settings` 是 `frozen` 的，**只能 `replace()`**，不能就地改 |

---

## 补充：开了 Reflection，为什么还要专门测两份草稿

只问一道题，有时原 Agent 本来就答对了，没法看出检查员有没有起作用。因此 `run_evaluation()` 在统计结果之前，按 `app.settings.reflection_enabled` 决定要不要再考两题。

两题共用固定材料：用户要求计算 `(12+8)*3`，工具结果明确是 `60`。这些是故意准备的虚构材料，不是重新执行了一次计算器；检查和重写调用的仍是真实千问，会产生费用。

| 草稿 | 期望 | 实际检查什么 |
| --- | --- | --- |
| `计算结果是50。` | `status="revised"`，答案有 `60` 没有 `50` | 能不能对照工具结果改错 |
| `已使用计算器，计算结果是60。` | `status="passed"`，答案原样保留 | 会不会把正确答案也乱改 |

`SYSTEM_PROMPT` 与 `AgentApp` 一起在文件顶部导入；这个模块本来就要加载，常量不需要另等开关。`reflect_answer` 则保留在开关分支里按需导入。

代码里的 `for name, draft, expected_status in [...]` 是一次取出三个值：题名、草稿、期望状态。每次调用 `reflect_answer()`，拿到选用答案、检查记录和原始模型消息；这两道独立样例只用前两项判断通过与否，再追加到原来的 `results`。

这两题不写新的聊天正文，只把结果留在评测报告；原有聊天题仍通过 `app.chat()` 测整条流程。只看状态还不够，所以还核对实际答案；对稿还要检查 `answer == draft`。

工具统计里的 `if item.get("kind") != "reflection"` 也有用：内部检查不算模型调用了一个新工具。用量已经移到独立的 `usage_json` 列，不在工具轨迹里。没有 `kind` 的老记录依然计入工具列表。

它们只是两道小题，不能证明所有回答都改善。关闭开关不会执行这两题；开启后原有问答题也会经过检查，因此整轮评测的模型用量会增加。

## 本篇要带走的三句话

1. **`evaluate.py` 和 `pytest` 是两套互补的东西**——`pytest` 快、免费、测零件（模拟外部服务）；`evaluate.py` 慢、花钱、测整机（真调模型、真写数据库）。**零件全合格，装起来仍可能不转，所以两套都要有。**
2. **隔离靠 `replace()` 造一份新配置**——换 `user_id` 让数据不串、换 `data_dir` 让文件不串。因为 `Settings` 是 `frozen=True` 的，**"要改就复印一份"** 是唯一也最干净的做法。
3. **测试的两条关键纪律**：**"准备动作"和"检查动作"要分开**（准备用 `app.chat` / `save_turn`，检查用 `ask`），**并且每个断言都要想一想"集合为空时会不会空洞通过"**——这就是 `bool(name_keys)` 存在的理由，也是 `try/except` 让一道题失败不影响其他题的价值所在。

继续读：[agent.py：总指挥——十八般兵器怎么串成一轮问答](D:/study/AI/ai-agent-demo-study/docs/code-guide/agent.md)


## 报错去哪里查

命令行入口先调用 `setup_logging()`。单题和整场评测捕获到的异常用 `logger.exception()` 保存完整堆栈到 `data/logs/app.log`；报告和终端失败摘要仍使用简短提示。详见 [logging_config.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/logging_config.md)。
