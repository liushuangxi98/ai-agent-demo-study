# tools.py：给失忆天才准备的工具箱（四件工具）

对应源码：[agent_demo/tools.py](D:/study/AI/ai-agent-demo-study/agent_demo/tools.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

大模型只会"说"，不会"做"。它可以说"我来算一下 (12+8)*3"，但这句话本身一个数都算不出来；它也永远不知道"现在几点"，也不能自己打开你的知识库。

`tools.py` 就是**给模型配的工具箱**，里面四件工具：

| 工具 | 干嘛 | 为什么需要 |
| --- | --- | --- |
| `calculator` | 算数学式子 | 模型心算容易翻车 |
| `current_time` | 查电脑当前时间 | 模型的"时间感"停在训练那天 |
| `knowledge_search` | 搜知识库 | 让它能自己再去翻资料 |
| `delete_knowledge` | 删一份知识文档 | 演示破坏性操作必须人工批准 |

文件里只有一个对外函数 `build_tools()`，四个工具函数全写在它**肚子里**。

## 1. 先看懂一轮"递单子"的完整来回

```text
用户：计算 (12+8)*3
    → 模型递单子：请调用 calculator，expression="(12+8)*3"
    → LangChain 按单子找到 Python 函数，真的执行
    → 函数返回字符串 "计算结果：60"
    → 结果塞回给模型，模型看一眼，写最终回答
```

关键认知：**模型递单子 ≠ 单子被执行了**。执行是 Python 的事，"递单-执行-回单"的循环由 agent.py 交给 LangChain 管理。本文件只负责两件：把工具造出来、把每件事真正干好。

## 2. `build_tools()`：工具是怎么"带装备"出厂的

```python
def build_tools(knowledge):
    # 肚子里定义四个工具函数
    return [calculator, current_time, knowledge_search, delete_knowledge]
```

只传进一个参数：知识库管理员。最后一行只是**把四个工具装进列表交出去**，没有执行任何一个。

### `@tool`：给普通函数贴上"说明书"

```python
@tool
def knowledge_search(query: str) -> str:
    """检索知识库；本轮已自动检索，资料不足时可换关键词补充检索。"""
    return knowledge.context(query)
```

`@tool` 是**装饰器**，可以想象成"包装贴纸"：贴上去之后，这个普通 Python 函数被包成一个工具对象，自带三样东西——

1. 名字（函数名 `knowledge_search`）；
2. 参数结构（`query: str`，从类型标注提取）；
3. 说明书（三引号 docstring，**这段是给模型看的**，模型靠它判断"什么情况该用这件工具"）。

所以 docstring 在这里不是写给人看的注释，**是 prompt 的一部分**。写得含糊，模型就用得含糊。

包好的工具用 `.invoke({"query": "课程退款条件"})` 真正执行，字典的键必须和参数名对上。

### 为什么把四个函数写在 `build_tools` 肚子里？（闭包）

知识检索需要外层传进来的 `knowledge`。内层函数能记住这个对象，这就叫**闭包**：工具出厂时已经知道该去哪个知识库找资料，模型只填写 `query`。

计算器和时间工具不需要知识库，但仍放在同一个工厂函数里，方便对照四个工具怎样注册。

聊天记录和共享画像由 `agent.py` 在每轮开始时读取，不通过工具查询。

## 3. `calculator()`：全项目最讲安全的一个函数

模型传进来一个字符串 `expression`，比如 `"(12+8)*3"`。最省事的写法是直接 `eval(expression)`——**但这是大忌**：eval 会执行任意 Python 代码，要是递进来的式子是 `__import__('os').system('删库')`，就真的执行了。

所以这个函数分四步"先安检，再放行"：

### 第一步：把式子拆开看骨架

```python
tree = ast.parse(expression, mode="eval")
```

`ast` = Abstract Syntax Tree（抽象语法树），可以理解成"**把代码拆成零件清单，只看结构、不执行**"。`(12+8)*3` 拆出来是：数字、数字、加法、乘法几种零件；`open("x")` 拆出来会含"函数调用"零件——一眼就能认出危险分子。`mode="eval"` 要求整个输入必须是个表达式（一条式子，不是语句块）。

### 第二步：零件逐个过安检

```python
allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ...)
for node in ast.walk(tree):
    if type(node) not in allowed_nodes:
        raise ValueError("仅支持数字、括号、+ - * / // %")
    if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
        raise ValueError("仅支持数字、括号、+ - * / // %")
```

`ast.walk` 把树上每个零件挨个拿起来对准入名单。名单里只有：数字、括号、正负号、`+ - * / // %` 六种运算。**函数调用、变量名、属性访问、幂运算 `**` 全不在名单上**，见一个毙一个。

第二个 if 再卡数字的种类：只许整数和小数。用 `type(...) not in (int, float)` 精确判断而不是 `isinstance`，是因为 Python 里 `True` 也是 int 的"亲戚"（布尔继承自整数），宽判断会把 `True+1=2` 这种放进来的。

### 第三步：过检的才准计算

```python
result = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
```

| 零件 | 白话 |
| --- | --- |
| `compile(tree, ...)` | 把检查过的结构准备成可执行形式 |
| `"<calculator>"` | 给这段代码起个名字方便报错定位，不读文件 |
| 第二个参数 `{"__builtins__": {}}` | **把 Python 的内置函数全断电**：里面没有 print、没有 open、没有 __import__，就算漏进个变量名也无处借力 |
| 第三个参数 `{}` | 也没有任何外部变量可用 |

双保险思路：白名单已排除危险结构，再给 eval 一个"什么都没有的房间"。

### 第四步：验结果、报答案

```python
if not math.isfinite(result) or abs(result) > 1e12:
    raise ValueError("结果超出教学计算器范围")
return f"计算结果：{result:.12g}"
```

`isfinite` 挡无穷大/NaN（比如 `1e308*10`），`1e12`（一万亿）封顶防天文数字刷屏。`.12g` 最多 12 位有效数字。

最后整个函数包在 `try/except (ValueError, SyntaxError, ArithmeticError)` 里，任何失败都变成一句 `"计算失败：原因"` 的**普通字符串**返回。为什么不当场炸掉？因为**模型需要看到失败原因**——它看到"计算失败：仅支持数字、括号…"，会调整式子再试。工具报错也是对话的一部分。

| 输入 | 结局 |
| --- | --- |
| `(12+8)*3` | `计算结果：60` |
| `10//3` | `计算结果：3`（整除）|
| `1/0` | `计算失败：division by zero` |
| `2**100` | 被名单毙掉（不支持幂）|
| `open("x")` | 被名单毙掉（函数调用）|

## 4. `current_time()`：问电脑，不是问模型

```python
now = datetime.now().astimezone()
return f"当前时间：{now:%Y-%m-%d %H:%M:%S %z}"
```

`datetime.now()` 取本机当前时间，`.astimezone()` 给它挂上"我在哪个时区"的标签，`%z` 显示成 `+0800` 这样的偏移。没有单独配时区——换台电脑跑，就是那台电脑的时间。模型自己"报时间"是拿训练截止日的旧记忆瞎猜，这个工具就是堵这个漏的。

## 5. `knowledge_search()`：一句话转交给图书馆

```python
return knowledge.context(query)
```

自己不干活，把检索词递给 `rag.py` 的 `context()`：问题变向量 → Chroma 找最近片段 → 带来源排好返回。

背景知识：每轮问答 `agent.py` 会**直接调用 `knowledge.context()` 自动检索一次**（轨迹里仍用 `knowledge_search` 标记，并带 `trigger: automatic`），这件工具的存在是给模型**补查**用的——自动检索那批资料不够时，模型可以换个关键词再捞一轮。docstring 里那句"资料不足时可换关键词补充"就是专门写给模型看的 Usage 指南。

## 6. `delete_knowledge()`：删知识文档，但要人批准

```python
@tool
def delete_knowledge(source: str) -> str:
    """删除一份知识文档；需要人工批准。source 用来源名，如 course.md 或 uploads/notes.txt。"""
    return knowledge.delete(source)
```

它只把来源名转交给 `rag.py` 的 `KnowledgeBase.delete()`；真正删原件、同步索引在那边。

**审批不是这个工具自己做的**。`agent.py` 给 Agent 挂了 `HumanInTheLoopMiddleware`：模型一请求 `delete_knowledge`，图就暂停，网页弹出批准/拒绝；批准后工具才执行。权限表、钩子和恢复流程见 [tool_policy.md](D:/study/AI/ai-agent-demo-study/docs/code-guide/tool_policy.md)。

## 7. 走一遍："现在几点？顺便算下 2 月的天数除法 28/7"

| 步 | 发生什么 |
| --- | --- |
| 1 | 模型读完问题，递两张单子：current_time（无参）和 calculator("28/7") |
| 2 | LangChain 挨个执行，得到"当前时间：…"和"计算结果：4.0" |
| 3 | 结果都塞回给模型，模型合成一句人话回答 |
| 4 | 这两次动作进了轨迹，网页"工具调用轨迹"里能看到 |

一轮调几个工具、先调哪个，全是模型自己临场决定的——这就是"Agent"和"固定脚本"的区别。

## 本篇要带走的四句话

1. 模型只会递单子，真正干活的是这里的 Python 函数；
2. 知识检索工具记住外层的知识库对象，模型只需填写检索词；
3. calculator 的"先拆结构、过白名单、再断电执行"是处理一切"用户输入的代码"的通用安全模板，值得抄。
4. 删除这类破坏性工具只负责干活，审批由 `tool_policy.py` 的 HITL 中间件在更外层负责。

继续读：[documents.py：仓库前台——收件、验货、抠文字](D:/study/AI/ai-agent-demo-study/docs/code-guide/documents.md)。
