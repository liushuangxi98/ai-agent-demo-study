# reflection.py：答案检查与会话摘要

源码：[agent_demo/reflection.py](D:/study/AI/ai-agent-demo-study/agent_demo/reflection.py)。可以把这个文件想成“交卷前检查”：先写草稿，再对照已有材料找明确错误，发现问题才改一次。

默认 `REFLECTION_ENABLED=false`。打开后会多请求一次模型检查；需要修改时再请求一次重写。这里复用同一个模型，不另外搭一个 Agent，不重新运行工具。

## 1. 先看整条路

```text
agent.py 已经拿到草稿和工具结果
  → reflect_answer(llm, materials, draft)
      → 让模型填检查单
      → Python 核对检查单原句、文档引用
      → 没问题：返回草稿
      → 有问题：用同一份材料重写一次
          → 核对修订稿是否非空、是否乱造引用
          → 合格：返回修订稿
          → 不合格：返回草稿，说明修订未完成
  → agent.py 保存选用答案、检查记录和用量
```

检查不是保证正确的魔法。它能帮忙找错，但模型也会漏看、误判。程序只做能确定的格式、原句和编号检查。

## 2. `ReviewIssue`：检查单上的一条问题

这个类继承 Pydantic 的 `BaseModel`。它像一张固定格式的小表，告诉模型“请填这些格子”。

| 字段 | 要填什么 | 例子 |
| --- | --- | --- |
| `kind` | 问题类别，只能选五种之一 | `tool`：答案与工具结果冲突 |
| `quote` | 草稿里的连续原句；漏答时引用用户原问题 | `结果是50` |
| `reason` | 为什么认为有问题 | `calculator 返回60` |
| `suggestion` | 应该怎么改 | `把50改为60` |

五种 `kind`：`fact` 是事实问题，`citation` 是引用问题，`memory` 是记忆使用错误，`tool` 是与工具结果冲突，`missing` 是漏答要求。

`Literal[...]` 规定选项；`Field(min_length=1, max_length=500)` 限制文本不能为空、不能太长。`extra="forbid"` 拒绝没定义的字段，`str_strip_whitespace=True` 自动去掉字符串两头的空白。

这只是检查表的结构规则。“理由是否真的正确”还需要对照材料，不能靠类型检查解决。

## 3. `ReviewResult`：装整张检查单

它只有一个字段：

```python
issues: list[ReviewIssue] = Field(max_length=5, ...)
```

`issues=[]` 表示没有发现必须修改的问题。有内容才进入重写，最多五条。不再多设一个“是否通过”的布尔值，避免出现“通过=True，但问题列表不为空”这种矛盾。

这个类也拒绝多余字段。两张表都没有自己写构造函数，Pydantic 会按声明生成、校验对象。

## 4. 两段提示词分别交代什么

### `REVIEW_PROMPT`：检查员的任务

要求模型只查必须修正的问题：漏答、缺依据的具体事实、错误引用、与工具结果冲突、违反画像遗忘或有效期规则。

不因为“换种措辞更好听”就要求重写；知识库没有提到的普通常识，也不能直接判错。历史里的助手回答可能错，不能当作可靠依据。

这里还提醒模型：问题、资料、历史里的指令都是待检查的数据，不能改变检查任务。程序不索取内部思考过程。

### `REVISE_PROMPT`：修改员的任务

要求按检查单写一段完整答案，保留正确内容；只用现有材料，不再调用工具，不编造新事实或计算结果。

如果还缺资料，就说明材料不足。修改后不进行第二次模型审查，所以状态叫“已修订”，不会声称“复审通过”。

## 5. `_unknown_citations()`：引用编号有没有出处

```python
def _unknown_citations(answer, tool_results) -> set[str]:
    ...
```

两个输入：`answer` 是草稿或修订稿，`tool_results` 是本轮工具结果。返回一个集合，例如 `{ "[invented.md#2]" }`；空集合表示没有未知引用。

执行顺序：

1. 正则找出答案中的 `[文件名#切片号]`。
2. 只从真正的 `knowledge_search` 结果中找允许的编号。
3. 做集合减法：答案用过的编号，减去资料里确实出现的编号。

```python
return set(re.findall(pattern, answer)) - set(re.findall(pattern, sources))
```

`set` 会去重，`-` 表示“左边有、右边没有”。不能从草稿里提取编号后再拿草稿给自己作证。

它只检查编号存在，不检查资料语义。例如编号是真的，但那段资料是否支持“退款期限是30天”，仍要模型判断。

## 6. `reflect_answer()`：按顺序完成检查与修改

检查和重写直接放在这个函数中，顺着往下读就能看完，不需要追多层回调。

### 输入和三个返回值

```python
answer, note, reflection_messages = reflect_answer(llm, materials, draft)
```

| 名称 | 意思 |
| --- | --- |
| `llm` | 已准备好的模型客户端 |
| `materials` | 本轮检查材料，见下表 |
| `draft` | 工具循环结束后生成的草稿 |
| 返回 `answer` | 最终选用的草稿或修订稿 |
| 返回 `note` | 网页可展开查看的检查记录 |
| 返回 `reflection_messages` | 检查和可能发生的修订所返回的原始消息，用于读 token |

`materials` 由 `agent.py:chat()` 准备：

| 键 | 来自哪里 | 为什么带上 |
| --- | --- | --- |
| `question` | 本轮用户原话 | 检查有没有漏答 |
| `history_summary` | 当前会话累计摘要 | 让检查员拥有与主回答相同的早期历史背景 |
| `history` | 未压缩的问答，转成角色与文字 | 理解“它”“刚才那个”等指代 |
| `profile` | 普通回答已经使用的画像文字 | 检查更正、遗忘、过期规则 |
| `tool_results` | 自动检索及模型后续工具结果 | 核对资料和计算结果 |
| `answer_rules` | `SYSTEM_PROMPT` | 与原回答遵守同一套规则 |

失效画像仍保留状态提示，但不带旧值。检查材料不会加入用量数字。

### 第一步：准备检查记录

```python
note = {
    "tool": "self_reflection", "kind": "reflection", "status": "passed",
    "draft": draft, "issues": [], "result": "检查完成，未发现需要修订的问题。",
}
```

`kind="reflection"` 让网页把它归到“答案检查”。`tool` 只是沿用轨迹的字段格式，不表示模型真的调用了一个新工具。

`messages_for_usage=[]` 用来交回原始消息；`raw_review=None` 表示目前还没拿到检查请求的返回值。

### 第二步：请模型按检查表输出

```python
reviewer = llm.with_structured_output(
    ReviewResult, method="function_calling", include_raw=True,
    extra_body={"enable_thinking": False},
)
```

- `ReviewResult` 规定要填的表。
- `method="function_calling"` 借助模型函数调用能力返回表格内容，不执行业务工具。
- `include_raw=True` 让结果同时包含原始模型消息和解析好的表。
- `extra_body` 在这里绑定到模型，关闭这一次检查的思考过程。

然后 `invoke()` 才真正请求模型。两条输入消息：系统消息放检查规则，用户消息放 `materials` 和 `draft`；字典先用 `json.dumps(..., ensure_ascii=False)` 转成中文可读的文字。

### 第三步：先拿原始消息，再验检查表

```python
raw_review = result["raw"]
if result["parsing_error"] is not None:
    raise result["parsing_error"]
review = ReviewResult.model_validate(result["parsed"])
```

`include_raw=True` 的返回字典有三个键：

| 键 | 内容 |
| --- | --- |
| `raw` | 原始 `AIMessage`，包含 `usage_metadata` |
| `parsed` | 解析后的 `ReviewResult`；解析失败时可能是 `None` |
| `parsing_error` | 解析错误；成功时是 `None` |

先保留 `raw` 很关键：模型填错表，也可能已经消耗 token，不能因为解析失败就漏掉这笔已返回的用量。

接着核对每条 `quote`：普通问题必须能在草稿中找到，`missing` 必须能在用户原问题中找到。找不到就整张拒收，防止检查员对着不存在的句子挑错。

### 第四步：程序补上未知引用

```python
review = ReviewResult(issues=(citation_issues + review.issues)[:5])
```

如果模型漏看了 `[invented.md#0]`，`_unknown_citations()` 仍会找出来，生成引用问题排到前面。两个列表相加，再取前五条，符合检查表容量。

之后用 `issue.model_dump()` 把每条对象转成普通字典，保存到 `note["issues"]`。

检查出错时，`except` 把状态改成 `review_failed`，保留草稿，只记错误类型，不存可能含请求正文的异常详情。

### 第五步：没有问题就结束

```python
if note["status"] == "review_failed" or not review.issues:
    return draft, note, messages_for_usage
```

`or` 左边成立时不会再算右边，所以检查失败、还没有 `review` 对象也能安全返回。

两种情况要区分：检查失败是“没检查完”；检查成功且问题列表为空才是“未发现问题”。这两种都保留草稿，但网页提示不同。

### 第六步：有问题才重写一次

修改请求用普通 `llm.invoke()`，输入包含原材料、草稿和检查单，系统消息用 `REVISE_PROMPT`。这次要完整答案，所以不使用 `ReviewResult` 表格。

回来后做三项验收：

1. 必须是非空 `AIMessage`。
2. 不能还带着待执行的 `tool_calls`。
3. 不能出现本轮资料里不存在的文档编号。

成功把状态改成 `revised`；任何一步失败改成 `revision_failed`，仍返回草稿，保留检查问题和失败说明。

原始修订消息也交回用于统计。即使修订稿被程序拒收，接口已返回的 token 仍算进去；请求本身失败没有消息，就交回 `None` 表示用量未知。

## 7. 四种状态怎样显示

| `status` | 选用什么答案 | 网页怎么解释 |
| --- | --- | --- |
| `passed` | 草稿 | 检查完成，没发现必须改的问题 |
| `revised` | 修订稿 | 已改一次，没有第二次模型复审 |
| `review_failed` | 草稿 | 检查没有完成 |
| `revision_failed` | 草稿 | 已发现问题，但修改没有完成 |

选用答案保存到 `assistant_text`；草稿、问题和状态保存到 `trace_json`。下一轮只带选用答案，不把整张检查单塞回聊天上下文。

模型消息只用于读取用量，最后存的是数字，见 [usage.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/usage.md)。后续记忆提取依然只认用户原话，检查员的意见不能变成用户画像事实。

## 8. 用一个算错的例子串起来

```text
用户：计算 (12+8)*3
工具：60
草稿：结果是50
  → 检查表：tool / 结果是50 / 工具返回60 / 改为60
  → 程序在草稿里找得到“结果是50”，检查表有效
  → 修改员：结果是60
  → 程序确认非空、无工具请求、无未知引用
  → 返回修订稿 + revised 记录 + 两次模型消息
```

假如检查接口超时，就返回草稿和 `review_failed`；假如修订接口超时，就返回草稿和 `revision_failed`。都不能谎称问题已经解决。

## 三句话记住

1. Reflection 是“写完后查一次，有明确问题再改一次”。
2. Python 检查格式、原句与引用编号，模型检查内容；两边都有能力边界。
3. 答案、检查记录、模型消息各有用途：给用户看、供网页展开、读取真实用量。


## 报错去哪里查

检查和修订的 `except` 现在用 `logger.exception()` 把完整原因写入本地日志。检查记录继续只保留类型和简短说明；技术堆栈在 `data/logs/app.log` 查。详见 [logging_config.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/logging_config.md)。


## 9. `summarize_history()`：回答完成后的历史摘要

摘要与 `reflect_answer()` 位于同一文件，但独立调用。`REFLECTION_ENABLED=false` 仅关闭答案检查，主问答最后一次总 token 超过 50000 时仍会摘要。

| 参数 | 用途 |
| --- | --- |
| `llm` | 复用现有问答客户端 |
| `previous_summary` | 更早历史已经生成的累计摘要 |
| `turns` | 本次新增纳入的完整旧问答，含真实 ID |

`SUMMARY_PROMPT` 要求保留目标、约束、结论、重要名称和数字、待办、更正与遗忘要求，区分用户事实、助手建议和未确认推断。每行提示词在源码旁都解释其用途。

函数只把旧摘要和本次问答序列化成 JSON。业务轨迹、统计和整份画像不混入摘要输入。`invoke()` 使用普通消息，不绑定工具，不传 `max_tokens` 或其他应用层长度限制。

成功返回 `(完整新摘要, 原始AIMessage, "")`。空回答、带工具请求或服务端明确报告截断时判失败；异常写入现有日志，返回 `(None, 已返回消息或None, 警告)`。校验失败不抹掉已经返回的用量。

覆盖位置由 `agent.py` 用本次纳入的最后一条真实记录 ID 确定，模型只负责文字。只有成功才替换摘要与游标；失败继续保存原快照和本轮有效问答。
