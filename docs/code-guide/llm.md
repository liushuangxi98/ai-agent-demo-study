# llm.py：一部装好号码和密钥的电话

对应源码：[agent_demo/llm.py](D:/study/AI/ai-agent-demo-study/agent_demo/llm.py) · [返回导航](D:/study/AI/ai-agent-demo-study/docs/code-guide/README.md)

## 一句话定位

整个文件只有一个函数 `build_llm()`，就干一件事：**造一部"电话机"**——号码（模型名）、接线地址（base_url）、门禁卡（api_key）都预先插好。以后谁要问千问问题，拿起这部电话拨号（`.invoke()`）就行。

**造电话 ≠ 打电话**。这个函数执行完，一个网络请求都还没发出去。为什么每个地方都要用同一部电话？要是每个文件调模型前都重新填一遍密钥地址，早晚有人漏填、填得不一致。集中造一次，到处复用。

## 它在项目里的位置

```text
config.py 装好 Settings 袋子
    → agent.py 调用 build_llm(settings)
    → 得到的电话机存进 self.llm
    → 问答用它、可选的答案检查用它、提取记忆也用它
```

## 1. `build_llm()` 逐行读

```python
def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=None,
        timeout=180,
        max_retries=2,
        use_responses_api=False,
        streaming=True,
    )
```

参数是从袋子里取三样：找哪个模型、凭什么访问、去哪个地址找。返回一个客户端对象——**不是模型的回答文字**。

### 类名叫 ChatOpenAI，打的却是阿里云？

对，这是新手第一大困惑点。`ChatOpenAI` 只是 LangChain 里的类名，**请求真正发给谁，由 `base_url` 决定**。阿里云百炼提供了一个"长得和 OpenAI 一样"的兼容接口（请求格式互相通用），所以拿 OpenAI 的客户端类 + 百炼的地址 = 给百炼打电话。国内各家模型（千问、DeepSeek、Kimi……）大多提供这种兼容口，一套代码轮流打给谁家都行。

### 每个参数白话过一遍

| 参数 | 白话 | 容易误会 |
| --- | --- | --- |
| `temperature=None` | 这栏不填，让服务按自己默认规则来 | **不等于温度=0**（0 表示每句都选最稳的词）；服务默认是多少就是多少 |
| `timeout=180` | 单次请求最多等 180 秒 | 不是"一轮聊天总共 180 秒"。一轮可能问模型好几次+提取记忆一次，每次各有自己的 180 秒 |
| `max_retries=2` | 碰上网络抖动这类可重试错误，额外再试 2 次 | 密钥错误、超限这类"硬失败"重试一万次也没用，这个参数救不了 |
| `use_responses_api=False` | 走 Chat Completions 这套请求格式 | 别真连到 OpenAI 官方地址去了——它家另有 Responses API 格式，百炼兼容口用的是老的那套 |
| `streaming=True` | 让模型以流式方式返回 | 本项目仍等待完整结果后再统一处理；流式主要利于底层连接与后续扩展 |

## 2. 电话打通后，回来的"信"长什么样

模型的回答装在一个叫 `AIMessage` 的"信封"里：

| 信封里的东西 | 怎么取 | 是什么 |
| --- | --- | --- |
| 文字 | `message.text` | 可能是给用户看的最终答案，也可能是"我先用个工具"的中间话 |
| 工具请求 | `message.tool_calls` | 模型填的单子："请帮我跑 calculator，参数 (12+8)*3" |

比如模型想算数时，`tool_calls` 里可能是：

```python
{"name": "calculator", "args": {"expression": "12+8"}, "id": "call_demo"}
```

**注意这只是"请求"，不是"结果"。** 模型自己没按计算器，它只是递了张单子。谁去执行？LangChain 框架的循环（`create_agent` 搭的工作流）接过单子、调用真正的 Python 函数、把结果塞回下一个信封。这段"递单-干活-回单"的循环在 agent.py，不在这里。

## 3. 为什么一部电话能干几种活

```text
同一个 self.llm
    ├── 问答：读问题+资料+历史，按需递工具单
    ├── 可选 Reflection：先按 ReviewResult 填检查单，有问题再改一次草稿
    └── 记忆提取：换个提示词，让它按 MemoryPatch 表格填"事实变化单"
```

电话机只是**通道**，每次拨号说什么内容由调用方现场组装。两次请求之间电话机不留任何内容——**它不存聊天、不存记忆**。模型能答"你叫小明"，是因为程序每轮从 MySQL 翻出画像、塞进这次的拨号内容里，不是电话机记性好。

这一条纠正一个常见错觉："AI 记得我" 的真相是 "程序每次都把资料重新递给失忆的天才看"。

## 4. `structured_memory.py` 里那个 `enable_thinking: False` 是什么

读那个文件会看到 `extractor.invoke(messages, extra_body={"enable_thinking": False})`。这是百炼接口的请求参数：这次调用不启用深度思考。现在 [reflection.py](D:/study/AI/ai-agent-demo-study/docs/code-guide/reflection.md) 的检查和重写也使用这个参数。它只影响对应请求，不会关闭原 Agent 的工具循环——同电话，不同任务配不同参数。

## 5. 模型用量为什么不挂在这个客户端上

`build_llm()` 返回的客户端会被应用复用，而用量要按会话、按轮分开，所以不在共享客户端上放“累计 token”属性。

模型本身已经返回 `AIMessage.usage_metadata`。`chat()` 直接读取本轮消息，反思和记忆提取用 `include_raw=True` 保留原消息，再把数字加到当前会话最后一轮的累计里，随问答保存。统计不需要回调，也不增加模型请求。

详见 [usage.py：直接加总模型返回值](D:/study/AI/ai-agent-demo-study/docs/code-guide/usage.md)。
