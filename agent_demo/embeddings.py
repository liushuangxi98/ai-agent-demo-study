"""创建向量化客户端：把文档片段和问题转换成可比较的数字向量。

Chroma 在入库和检索时调用它；本模块不直接保存向量。
"""

from langchain_openai import OpenAIEmbeddings

from agent_demo.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """参数：settings 提供百炼配置；返回文档与问题共用的向量化客户端。"""
    # 文档和问题必须用同一个客户端、同一个模型，向量才能落在同一坐标系。
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        request_timeout=30,
        max_retries=1,
        chunk_size=10,  # 每次请求最多 10 条文本；不是 rag.py 的切片长度。
        check_embedding_ctx_length=False,  # 片段已切好，交给服务端判断长度。
        model_kwargs={"encoding_format": "float"},
    )
