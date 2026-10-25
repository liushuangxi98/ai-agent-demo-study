"""测试共用的配置；只放虚构值，不读取 .env，也不连接真实服务。"""

import pytest

from agent_demo.config import Settings


@pytest.fixture
def settings():
    # 创建 Settings 现在会自动校验，所以测试也要提供格式合法的必填项。
    # .invalid 是供示例使用的域名；模型和数据库仍由各项测试替换成假的。
    return Settings(
        llm_api_key="test-key",
        llm_base_url="https://example.invalid/compatible-mode/v1",
        mysql_user="test-user",
        mysql_password="test-password",
    )
