"""创建问答模型客户端；真正请求发生在 invoke() 或 Agent 执行时。

客户端不保存长期记忆，历史由 memory.py 每轮读取后传入。
"""

from langchain_openai import ChatOpenAI

from agent_demo.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    """参数：settings 提供模型名、密钥与地址；返回 ChatOpenAI 客户端。

    这里只创建客户端，不发送请求；实际服务商由 base_url 决定。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=None,  # 不干预，使用服务端默认值。
        timeout=180,  # 单次 HTTP 请求的超时，不是整轮问答的总时长。
        max_retries=2,  # 连接失败、限流或服务端 5xx 时再试 2 次。
        # 百炼兼容接口使用 Chat Completions 协议，因此关闭 Responses API。
        use_responses_api=False,
        streaming=True,
    )
