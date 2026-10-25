# usage.py：把模型返回的数字加起来

源码：[agent_demo/usage.py](D:/study/AI/ai-agent-demo-study/agent_demo/usage.py)。这个文件只有一个函数 `summarize_usage()`，没有统计类，也不用给模型挂回调。

可以把它看成一个计算器：**拿到本轮模型返回的数字，加到上轮保存的累计里，再把两份数字交回去。**

## 1. 一轮怎么走

```text
chat() 记下开始时间
  → context(session_id) 同时取回近期聊天和上轮累计
  → Agent 跑完，拿到本轮所有 AIMessage
  → 开了反思：检查和修订也把模型消息交回
  → 提取记忆：交回变化、模型消息和警告
  → 回答完成后按最后一次主问答总 token 判断摘要，需要时追加摘要消息
  → summarize_usage(模型消息, 上轮累计, 到现在经过的秒数)
  → save_turn(..., usage=usage) 随问答一次存库
  → 网页读取 current 显示本轮，读取最后一行 total 显示累计
```

没有为了统计再调一次模型。也没有在网页刷新时重复相加。

## 2. 输入的三个参数是什么

```python
def summarize_usage(messages, previous, chat_seconds):
    ...
```

| 参数 | 装什么 | 例子 |
| --- | --- | --- |
| `messages` | 本轮每次模型请求返回的原始消息 | 工具调用请求、最终回答、检查、修订、记忆提取、摘要 |
| `previous` | 当前会话最后一轮保存的 `total` | 新会话没有记录，就用 `{}` |
| `chat_seconds` | 进入 `chat()` 到保存前经过多少秒 | `7.5` |

统计只用 `messages` 里的 `usage_metadata`。消息正文不会被这个函数另外保存。

**不要只拿最后一条回答的用量。** 模型可能先说“我要调计算器”，拿到结果后又请求一次。两次请求都消耗 token，都需要算。

也不要把旧历史的 AIMessage 再算一次。`agent.py` 先切出 `new_messages`，再挑本轮的 AIMessage；旧轮次已经包含在 `previous` 里。

## 3. 第一步：从每条消息抄回五个数字

```python
usage = (message.usage_metadata or {}) if isinstance(message, AIMessage) else {}
```

`isinstance()` 先确认这确实是模型消息。请求失败没有消息时，调用方交回 `None`，这里就用空字典。

模型返回的字段与存库字段这样对应：

| 存库名称 | 从哪里取 | 意思 |
| --- | --- | --- |
| `input_tokens` | `usage_metadata["input_tokens"]` | 这次模型请求的输入数量 |
| `output_tokens` | `usage_metadata["output_tokens"]` | 这次模型请求的输出数量 |
| `total_tokens` | `usage_metadata["total_tokens"]` | 接口报告的总数量 |
| `cached_tokens` | `input_token_details.cache_read` | 输入里有多少命中了缓存 |
| `reasoning_tokens` | `output_token_details.reasoning` | 输出里有多少属于推理 token |

缓存 token 已经算在输入里，推理 token 已经算在输出里。**不能把这两项再加到总 token 上。** 缓存命中也不代表输入 token 数量变成零。

`.get()` 找不到字段时给 `None`。接口明确返回 `0` 才是零；没提供不能假装是零。

## 4. 第二步：把本轮各次请求加起来

```python
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")

for field in USAGE_FIELDS:
    values = [record[field] for record in records]
    current[field] = sum(values) if values and all(value is not None for value in values) else None
```

五项统计共用这一个字段元组，避免“本轮相加”和“累计相加”两处各写一遍、漏掉某一项。

比如本轮两次请求：

| 请求 | 输入 | 输出 | 总量 | 缓存 | 推理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 请求调用工具 | 1,577 | 31 | 1,608 | 0 | 15 |
| 看完工具后回答 | 1,635 | 104 | 1,739 | 1,024 | 42 |
| 本轮合计 | 3,212 | 135 | 3,347 | 1,024 | 57 |

开了反思，还要加检查和可能发生的修订；记忆提取也要加进去。上表只是两次问答请求的例子。

如果其中一笔没有缓存数字，就不能知道本轮完整的缓存数，所以这一项是 `None`。其他完整项目照常计算，不会因为缺缓存信息连输入总数也一起丢掉。

结构化输出用 `include_raw=True`，即使填表解析失败，原始消息仍能拿到，已返回的用量照算。如果请求本身失败、没有原始消息，就保留未知，不估算。

## 5. 第三步：上轮累计加本轮

```python
before = previous.get(field, 0)
total[field] = before + current[field] if before is not None and current[field] is not None else None
```

- 新会话：`previous={}`，每项从 `0` 开始。
- 老会话：直接用最后一轮的累计，不读全部历史相加。
- 上轮累计或本轮有一项未知：新的这一项累计仍是未知，不能把不完整小计说成总消耗。

然后加耗时、加轮数：

```python
current["chat_seconds"] = chat_seconds
total["chat_seconds"] = previous.get("chat_seconds", 0) + chat_seconds
total["turns"] = previous.get("turns", 0) + 1
```

`turns` 是这个会话已统计的问答轮数，工具循环调用五次模型也仍是一轮问答。

## 6. 返回什么，数据库存什么

```python
return {"current": current, "total": total}
```

`conversation_turns` 一行是一轮问答。独立的原生 JSON 列 `usage_json` 直接保存这个字典，例如第二轮：

```json
{
  "current": {
    "input_tokens": 900,
    "output_tokens": 90,
    "total_tokens": 990,
    "cached_tokens": 0,
    "reasoning_tokens": 0,
    "chat_seconds": 3.5
  },
  "total": {
    "input_tokens": 1000,
    "output_tokens": 100,
    "total_tokens": 1100,
    "cached_tokens": 100,
    "reasoning_tokens": 10,
    "chat_seconds": 6.0,
    "turns": 2
  }
}
```

这意味着第一轮消耗输入 100、输出 10、用时 2.5 秒。第二轮只需要这份上轮累计，不用再把第一轮正文翻一遍。

`trace_json` 继续只放工具与反思。统计不会变成反思的参考证据，也不进入用户画像或下一轮提示词。

## 7. 网页怎么显示

`app.py:render_chat()` 读取当前会话最近 50 轮：

```python
render_usage((row.get("usage_json") or {}).get("current"))
```

每条回答显示本轮。底部直接拿最后一行：

```python
latest_usage = (rows[-1].get("usage_json") or {}) if rows else {}
total = latest_usage.get("total")
render_usage(total, prefix="累计")
```

即使会话有 100 轮、页面只展示最近 50 轮，最后一行的 `total` 也已经包含前 100 轮。

百分比由 `render_usage()` 现算：

- 缓存命中率 = 缓存 token ÷ 输入 token。
- 推理 token 占比 = 推理 token ÷ 输出 token。
- 累计比例用累计数量算，不能平均每轮的百分比。

刷新只读库；切回旧会话就读旧会话的累计；新会话从零开始。没有记录显示“未记录”，字段缺失显示“未提供”，分母为零显示“—”。

## 8. 耗时的边界在哪里

`agent.py:chat()` 只在开始和保存前各读一次 `perf_counter()`。两次数值相减，就是本轮处理耗时。

它包含读取记忆、检索、模型与工具循环、可选反思、记忆提取、摘要及这些步骤中的等待；不包含最后的数据库保存和网页绘制。因此页面叫“处理耗时”，不是“纯模型耗时”。

本轮即使没有可记住的事实，记忆提取也发过请求，其 token 和等待时间仍计入。

## 9. 这个简版的使用范围

只累计当前用户的当前会话，只统计成功保存的问答；Embedding 与没有保存的失败问答不计入。请求内部重试未返回的用量无法凭返回值还原。

累计快照按同一会话顺序提问使用，不处理多个请求同时写入同一会话，也不自动重算被手工删除或修改过的历史账目。

本次按重新开始运行的方式调整表结构，不保留上一版迁移代码。`create_all()` 只建缺失表，不修改已有表；已有旧表需一次性重建，正常启动不会清空数据。具体操作见项目 README。

## 三句话记住

1. 模型已经返回用量，直接读 `usage_metadata`，不再安排回调记账员。
2. `current` 存本轮，`total` 存截至本轮的会话累计；下一轮复用最后一条累计。
3. 网页负责展示，数据库负责留存，`chat()` 只计整轮处理时间。
