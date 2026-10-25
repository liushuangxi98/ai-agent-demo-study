"""检查输入、消息协议和自动检索；真实模型与数据库链路由 evaluate.py 验证。"""

# ---------------------------------------------------------------------------
# 第三方库导入
# ---------------------------------------------------------------------------
# unittest.mock 是 Python 自带的“造假工具”：
#   Mock()  造一个假对象，随便调它的任何方法都不会报错，还能记录“被怎么调过”
#   ANY     在断言里表示“这个参数是什么都行，我不检查”
# 用它们就能在不连模型、不连数据库的情况下，单独测某个函数的逻辑。
from dataclasses import replace
from unittest.mock import ANY, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 注意这里 import 的是“模块本身”（as agent_module），
# 因为下面要替换模块里的某个名字（monkeypatch.setattr(agent_module, ...)）。
import agent_demo.agent as agent_module
from agent_demo.agent import AgentApp, collect_trace
from agent_demo.config import Settings
from agent_demo.memory import validate_session_id
from agent_demo.rag import KnowledgeBase
from agent_demo.tools import build_tools


# ---------------------------------------------------------------------------
# 计算器：正常输入要算对
# ---------------------------------------------------------------------------
# parametrize 让同一个函数跑三组数据，分别对应普通运算、负数除法、取余。
@pytest.mark.parametrize("expression, expected", [
    ("(12+8)*3", "60"), ("-9/2", "-4.5"), ("10%3", "1"),
])
def test_calculator(expression, expected):
    # 计算器不查询知识库，这里用假对象代替真实知识库。
    calculator = next(tool for tool in build_tools(Mock())
                      if tool.name == "calculator")

    # 工具对象用 .invoke({"参数名": 值}) 来执行；
    # 字典的键必须和函数参数名对上（这里是 expression）。
    assert calculator.invoke({"expression": expression}) == f"计算结果：{expected}"


# ---------------------------------------------------------------------------
# 计算器：危险或超范围的输入必须被拦住
# ---------------------------------------------------------------------------
# 这一组全是“本该被拒绝”的：
#   __import__(...)  试图执行任意代码（最危险的那个）
#   2**100           幂运算不在白名单里
#   True+1           布尔值伪装成整数，必须被精确类型检查挡掉
#   1/0              除零
#   1e99             结果超出教学计算器的上限
@pytest.mark.parametrize("expression", [
    "__import__('os').getcwd()", "2**100", "True+1", "1/0", "1e99",
])
def test_calculator_rejects_unsupported_input(expression):
    calculator = next(tool for tool in build_tools(Mock())
                      if tool.name == "calculator")

    # 重点：失败时不是抛异常，而是返回一句“计算失败：原因”的普通字符串。
    # 这样模型能看到原因并改用别的式子，而不是把整轮对话弄崩。
    assert calculator.invoke({"expression": expression}).startswith("计算失败：")


# ---------------------------------------------------------------------------
# 注册工具：剩余三件工具能正常使用
# ---------------------------------------------------------------------------
def test_registered_tools_keep_knowledge_search_and_local_time():
    knowledge = Mock()
    knowledge.context.return_value = "[course.md#0]\n课程资料"
    tools = {tool.name: tool for tool in build_tools(knowledge)}

    assert set(tools) == {"calculator", "current_time", "knowledge_search"}
    assert tools["knowledge_search"].invoke({"query": "退款"}) == "[course.md#0]\n课程资料"
    knowledge.context.assert_called_once_with("退款")
    assert tools["current_time"].invoke({}).startswith("当前时间：")


# ---------------------------------------------------------------------------
# 会话 ID 的格式校验
# ---------------------------------------------------------------------------
def test_session_id():
    # 合法的原样返回。
    assert validate_session_id("study-01") == "study-01"

    # 带空格的非法，必须抛 ValueError。
    # with pytest.raises(...) 的意思是“下面这段代码必须抛这个异常，否则测试失败”。
    with pytest.raises(ValueError):
        validate_session_id("bad session")


# ---------------------------------------------------------------------------
# 用户 ID 的格式校验（注意这里连的是 Settings 的构造，不是单独的函数）
# ---------------------------------------------------------------------------
# 三种非法输入：空、含空格、超长（65 个字符）。
@pytest.mark.parametrize("user_id", ["", "bad user", "a" * 65])
def test_user_id_rejects_empty_or_invalid_identifiers(user_id):
    # 因为 Settings 是 @dataclass，__post_init__ 会在“创建对象的那一刻”自动跑校验，
    # 所以这里只要尝试构造，就相当于在测校验逻辑。
    # match="USER_ID" 表示错误信息里必须包含这两个字。
    with pytest.raises(ValueError, match="USER_ID"):
        Settings(user_id=user_id)


# ---------------------------------------------------------------------------
# 必填配置缺失时，只报配置名，不能把密钥打印出来
# ---------------------------------------------------------------------------
def test_missing_configuration_lists_names_only():
    # Settings() 全用默认值（密钥等都是空字符串），校验必然失败。
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        Settings()


def test_loading_settings_rejects_missing_required_fields(monkeypatch):
    """读取空配置时就报错，不再等到创建 AgentApp。"""
    from types import SimpleNamespace
    import agent_demo.config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", Mock())
    monkeypatch.setattr(config_module, "os", SimpleNamespace(environ={}))

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        config_module.load_settings()


@pytest.mark.parametrize("changes, error_name", [
    ({"llm_base_url": "bad-url"}, "LLM_BASE_URL"),
    ({"mysql_port": 0}, "MYSQL_PORT"),
    ({"mysql_port": 65536}, "MYSQL_PORT"),
])
def test_copied_settings_are_checked_automatically(settings, changes, error_name):
    """replace 也会创建新配置，错误的地址和端口不能绕过自动检查。"""
    with pytest.raises(ValueError, match=error_name):
        replace(settings, **changes)


def test_app_rejects_incomplete_settings_before_opening_resources(monkeypatch):
    """先创建配置，再创建应用；缺项在配置创建时拦住，不会打开模型资源。"""
    builder = Mock()
    monkeypatch.setattr(agent_module, "build_llm", builder)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        AgentApp(Settings())

    builder.assert_not_called()


# ---------------------------------------------------------------------------
# 工具轨迹的配对逻辑
# ---------------------------------------------------------------------------
def test_trace_matches_call_ids():
    # 手工拼两条消息：模型说要调 calculator（带 id "call1"），
    # 工具返回了结果（也带 id "call1"）。
    calls = [{"name": "calculator", "args": {"expression": "1+2"}, "id": "call1", "type": "tool_call"}]
    messages = [AIMessage(content="", tool_calls=calls), ToolMessage(content="3", tool_call_id="call1")]

    # 期望它们被按 id 正确配成一对，得到“工具名 + 参数 + 结果”。
    assert collect_trace(messages) == [{"tool": "calculator", "args": {"expression": "1+2"}, "result": "3"}]


# ---------------------------------------------------------------------------
# 一个有“半成品 app”的测试夹具
# ---------------------------------------------------------------------------
# @pytest.fixture 是装饰器，用来准备测试需要的前置条件。
# 测试函数只要把夹具名写进参数列表，pytest 就会自动调用它并把返回值传进来。
#
# 参数里有 monkeypatch 时，pytest 会自动注入这个内置工具 ——
# 它能临时替换掉某个模块里的名字，测试结束自动还原。
@pytest.fixture
def local_chat(monkeypatch, settings):
    # 只隔离外部服务，不添加离线运行模式；执行真实 chat() 和 @tool 包装函数。
    #
    # 重点技巧：__new__(AgentApp) 能“造出一个对象但不执行 __init__”。
    # 因为 __init__ 会去连数据库、建向量库，测试里不需要那些，
    # 所以跳过它，手动把三个成员填成假的。
    app = AgentApp.__new__(AgentApp)
    app.settings = settings
    app.llm = object()          # chat() 只把它转手交给 create_agent，内容无所谓
    app.knowledge = Mock()
    app.memory = Mock()

    # 给假对象预设“被调用时返回什么”。
    app.memory.profile.return_value = {}
    app.memory.context.return_value = ([{"id": 1, "user_text": "你好", "assistant_text": "你好"}], {}, {})  # 上下文现在交回原文行、累计和摘要。

    # 造一个假的“Agent 图”。side_effect 给它一个函数，每次调用都执行这个函数。
    # 这里模拟的是真实图的行为：把新消息追加到消息列表后面，
    # 并且最后一条是一条普通的 AI 回答（这样 chat() 的有效性检查才能通过）。
    graph = Mock()
    graph.invoke.side_effect = lambda state, config: {
        "messages": state["messages"] + [AIMessage(content="本地协议测试回答")]
    }

    # monkeypatch.setattr(模块, "名字", 新值)：把这个模块里的名字临时换掉。
    # 这里把 create_agent 换成“一个永远返回上面那个假图的假函数”，
    # 于是 chat() 里那一行就不会真的去装配 LangChain。
    # 注意函数名是 create_agent，所以 agent.py 里也必须是 create_agent（两边得对上）。
    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=graph))
    # 同样把“从画像提取事实”换成“永远返回空列表”，免得去调真模型。
    monkeypatch.setattr(agent_module, "extract_changes", Mock(return_value=([], None, "")))

    # yield 表示这个夹具交出一对值给测试用；测试跑完后会继续执行 yield 之后的代码。
    return app, graph


# ---------------------------------------------------------------------------
# 用户没写“请查知识库”，程序也要自动查，而且存库时要用原始问题
# ---------------------------------------------------------------------------
def test_chat_searches_without_a_user_hint_and_keeps_original_question(local_chat):
    app, graph = local_chat
    question = "我的伴侣叫什么名字？"
    # 虚构测试数据，不读取或发送用户的任何私人文档。
    evidence = "[uploads/profile.txt#0]\n测试人物的伴侣名字是晨星。"

    # 让假知识库返回一段假证据。
    app.knowledge.context.return_value = evidence
    result = app.chat("study-01", question)

    # 验证 1：知识库确实被调用了，而且用的就是用户原句。
    app.knowledge.context.assert_called_once_with(question)

    # 验证 2：发给模型的最后一条消息里，既要有问题，也要有检索到的证据。
    request = graph.invoke.call_args.args[0]
    assert [message.text for message in request["messages"][1:3]] == ["你好", "你好"]  # 画像之后才是成对的历史原文。
    assert question in request["messages"][-1].text
    assert evidence in request["messages"][-1].text

    # 验证 3：轨迹里第一条应该是“自动检索”，并标上 trigger: automatic。
    assert result["trace"] == [{
        "tool": "knowledge_search", "args": {"query": question},
        "result": evidence, "trigger": "automatic",
    }]

    # 验证 4：存进数据库的是“原始问题”，不是拼了证据的整段文本 ——
    # 否则每轮历史里都会塞一大段检索结果。ANY 表示时间参数不检查。
    app.memory.save_turn.assert_called_once_with(
        "study-01", question, result["answer"], result["trace"], [], ANY, usage=result["usage"], summary=None  # 无摘要时明确保存 SQL NULL。
    )


# ---------------------------------------------------------------------------
# 自动检索之后，模型自己调的工具也要被记进轨迹
# ---------------------------------------------------------------------------
def test_model_tool_trace_is_preserved_after_automatic_search(local_chat):
    app, graph = local_chat
    app.knowledge.context.return_value = "知识库未找到足够相关的证据。"

    # 这次让假图模拟一次“模型调用计算器”的完整来回：
    # 模型提出调用（带 id math1）→ 工具返回结果 → 模型给出最终回答。
    graph.invoke.side_effect = lambda state, config: {"messages": state["messages"] + [
        AIMessage(content="", tool_calls=[{
            "name": "calculator", "args": {"expression": "1+2"},
            "id": "math1", "type": "tool_call",
        }]),
        ToolMessage(content="计算结果：3", tool_call_id="math1"),
        AIMessage(content="3"),
    ]}
    result = app.chat("study-01", "计算 1+2")

    # 最后一条是普通回答，会被当成最终答案。
    assert result["answer"] == "3"

    # 轨迹里应当是“自动检索 + 模型自己调的计算器”，顺序也对得上。
    assert [item["tool"] for item in result["trace"]] == ["knowledge_search", "calculator"]
    assert result["trace"][1]["result"] == "计算结果：3"


# ---------------------------------------------------------------------------
# 检索失败时，不许“编一个回答”
# ---------------------------------------------------------------------------
def test_search_failure_does_not_produce_or_save_an_uninformed_answer(local_chat):
    app, graph = local_chat
    # side_effect 给的是一个异常对象：调用到这个方法就会把它抛出来。
    app.knowledge.context.side_effect = ConnectionError("模拟向量接口异常")

    # 检索挂了，整轮就该报错，而不是硬着头皮让模型瞎答。
    with pytest.raises(ConnectionError):
        app.chat("study-01", "我的伴侣叫什么名字？")

    # 后面的模型调用和保存都不该发生 —— 没回答就不该留记录。
    graph.invoke.assert_not_called()
    app.memory.save_turn.assert_not_called()


# ---------------------------------------------------------------------------
# 空问题要尽早拦住，别浪费一次检索和一次模型调用
# ---------------------------------------------------------------------------
def test_invalid_question_does_not_start_automatic_search(local_chat):
    app, graph = local_chat
    # 只输入一个空格，strip 之后就是空的。
    with pytest.raises(ValueError):
        app.chat("study-01", " ")

    app.knowledge.context.assert_not_called()
    graph.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# 长问题不该被悄悄截断
# ---------------------------------------------------------------------------
def test_long_question_has_no_arbitrary_character_limit(local_chat):
    app, graph = local_chat
    # 造一个 4004 字的超长问题。
    question = "测试问题" * 1001
    app.knowledge.context.return_value = "知识库未找到足够相关的证据。"
    app.chat("study-01", question)

    # 完整的问题必须原样出现在发给模型的消息里，一个字都不能少。
    assert question in graph.invoke.call_args.args[0]["messages"][-1].text


# ---------------------------------------------------------------------------
# 知识库没找到证据时，返回的应该是“未命中提示”，而不是让模型一律答不知道
# ---------------------------------------------------------------------------
def test_empty_knowledge_result_does_not_instruct_every_answer_to_be_unknown():
    # 又用了一次“造对象但跳过 __init__”的技巧：这里只测 context 一个方法，
    # 不需要真的去连向量库。
    knowledge = KnowledgeBase.__new__(KnowledgeBase)
    # 让检索永远返回空列表，模拟“什么都没找到”。
    knowledge.retrieve = Mock(return_value=[])

    # 注意返回的是“未找到证据”，而不是“必须回答不知道”。
    # 因为知识库没有资料时，模型仍然可以用常识或计算器等其他工具作答。
    assert knowledge.context("计算 1+2") == "知识库未找到足够相关的证据。"
