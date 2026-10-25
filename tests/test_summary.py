"""验证摘要触发、消息次序、覆盖边界与失败降级；不连接真实服务。"""

import json  # 检查真正发给摘要模型的材料。
from dataclasses import replace  # 为测试复制配置，不修改共享配置对象。
from types import SimpleNamespace  # 用虚构环境变量替代真实系统配置。
from unittest.mock import Mock  # 模型、数据库和知识库都用本地替身。

import pytest  # 提供夹具和参数化边界验证。
from langchain_core.messages import AIMessage, HumanMessage  # 使用真实消息类型检验 Agent 协议。

import agent_demo.agent as agent_module  # 替换 Agent 装配与事实提取入口。
import agent_demo.config as config_module  # 验证新配置入口且不读取本机 .env。
import agent_demo.reflection as reflection_module  # 单独验证摘要函数与答案检查的独立性。
from agent_demo.agent import AgentApp, SYSTEM_PROMPT  # 执行真实 chat 主流程。
from agent_demo.reflection import summarize_history  # 执行真实摘要请求与校验逻辑。


def model_reply(total=100, text="正常回答", **kwargs):  # 造一条携带完整用量的模型返回。
    """参数：总 token、正文和额外消息字段；返回真实 AIMessage。"""
    return AIMessage(  # 五项统计均提供，方便精确核对摘要费用。
        content=text, usage_metadata={"input_tokens": total - 5, "output_tokens": 5, "total_tokens": total, "input_token_details": {"cache_read": 0}, "output_token_details": {"reasoning": 0}},  # 输入加输出等于总量。
        **kwargs,  # 可传入服务端 finish_reason 等元数据。
    )  # 模型消息创建完成。


@pytest.fixture  # 每项测试各自创建替身，避免状态相互污染。
def summary_chat(monkeypatch, settings):  # 只隔离外部资源，保留真实业务流程。
    """返回应用、假 Agent 图和本轮模型消息列表。"""
    app = AgentApp.__new__(AgentApp)  # 避免构造时连接真实 MySQL 和 Chroma。
    app.settings = settings  # 使用默认 50000 阈值与保留 6 轮。
    app.llm = Mock()  # 摘要调用由此替身返回完整模型消息。
    app.llm.invoke.return_value = model_reply(110, "新的累计摘要")  # 摘要费用应计入本轮统计。
    app.memory = Mock()  # 记录读写参数，核对快照与用量。
    turns = [{"id": 10 + index * 3, "user_text": f"原问题{index}", "assistant_text": f"原回答{index}"} for index in range(8)]  # ID 故意不连续。
    app.memory.context.return_value = (turns, {}, {"text": "更早的摘要", "through_turn_id": 4})  # 8 轮尚未压缩原文。
    app.memory.profile.return_value = {}  # 默认无个人画像，专注验证会话历史。
    app.knowledge = Mock()  # 知识证据与摘要材料需要保持分界。
    app.knowledge.context.return_value = "[notes.md#0]\n本轮检索资料"  # 使用可追踪的虚构证据。
    outputs = [model_reply()]  # 测试可替换这组本轮新模型返回。
    graph = Mock()  # LangChain 图只由测试指定返回消息。
    graph.invoke.side_effect = lambda state, config: {"messages": state["messages"] + outputs}  # 输入消息原样回传，符合框架协议。
    monkeypatch.setattr(agent_module, "create_agent", Mock(return_value=graph))  # 不请求真实主模型。
    monkeypatch.setattr(agent_module, "extract_changes", Mock(return_value=([], model_reply(12), "")))  # 画像提取用量仍需累计。
    return app, graph, outputs  # 每项测试拿到可独立控制的三个对象。


@pytest.mark.parametrize("total,triggered", [(49999, False), (50000, False), (50001, True)])  # 精确检查严格大于的边界。
def test_summary_uses_last_main_total_and_strict_threshold(summary_chat, total, triggered):  # 不用输入峰值或整轮累计触发。
    app, graph, outputs = summary_chat  # 使用独立替身应用。
    outputs[:] = [model_reply(90000, "中间请求"), model_reply(total)]  # 早先请求消耗再大也不是最后一次返回。
    result = app.chat("summary-study", "当前问题")  # 执行真实判断与保存。
    assert app.llm.invoke.call_count == int(triggered)  # 最后一次总量超过 50000 才调用摘要。
    expected = 90000 + total + 12 + (110 if triggered else 0)  # 费用统计仍包含全部主模型与辅助请求。
    assert result["usage"]["current"]["total_tokens"] == expected  # 摘要触发信号与费用累计互不混淆。


def test_summary_preserves_tail_and_uses_real_database_cursor(summary_chat):  # 本轮回答也计入保留的 6 轮。
    app, graph, outputs = summary_chat  # 拿到 8 轮原文与旧摘要。
    turns, previous, old_summary = app.memory.context.return_value  # 保存原状态，核对函数不原地修改。
    outputs[:] = [model_reply(50001)]  # 最后一次主问答超过阈值。
    result = app.chat("summary-study", "当前用户原话")  # 9 轮中只压缩前 3 轮。
    payload = json.loads(app.llm.invoke.call_args.args[0][1].text)  # 查看真实交给摘要函数的 JSON。
    assert payload == {"previous_summary": "更早的摘要", "turns": turns[:3]}  # 只包含旧摘要与新纳入的旧问答。
    assert app.memory.save_turn.call_args.kwargs["summary"] == {"text": "新的累计摘要", "through_turn_id": 16}  # 真实 ID 16 不能替换成轮数 3。
    assert old_summary == {"text": "更早的摘要", "through_turn_id": 4}  # 旧快照保持原样。
    assert "当前用户原话" not in app.llm.invoke.call_args.args[0][1].text  # 本轮问答保留原文，不提前纳入摘要。
    assert "本轮检索资料" not in app.llm.invoke.call_args.args[0][1].text  # 检索材料不混成历史用户原话。
    assert agent_module.extract_changes.call_args.args[1] == "当前用户原话"  # 画像提取仍只接收用户原话。
    assert not result["memory_warning"]  # 摘要成功和提取成功时无警告。


def test_next_compression_merges_only_newly_covered_originals(summary_chat):  # 多次累计不得遗漏或重复纳入。
    app, graph, outputs = summary_chat  # 第一轮有 8 轮未压缩历史。
    turns = app.memory.context.return_value[0]  # 保存原文作为后续数据库返回的基础。
    outputs[:] = [model_reply(50001)]  # 两轮都触发正常摘要。
    app.chat("summary-study", "第一轮新问题")  # 先覆盖前 3 轮，游标到 ID 16。
    first_state = app.memory.save_turn.call_args.kwargs["summary"]  # 模拟第一轮已持久化的快照。
    tail = turns[3:] + [{"id": 100, "user_text": "第一轮新问题", "assistant_text": "正常回答"}]  # 数据库只返回旧游标后的 6 轮。
    app.memory.context.return_value = (tail, {}, first_state)  # 第二轮复用累计摘要。
    app.chat("summary-study", "第二轮新问题")  # 7 轮中只新增压缩最早一轮。
    payload = json.loads(app.llm.invoke.call_args.args[0][1].text)  # 查看第二次摘要输入。
    assert payload["previous_summary"] == "新的累计摘要"  # 继续总结旧摘要，不重新拉全部旧原文。
    assert payload["turns"] == tail[:1]  # 新增覆盖只有 ID 19 的一轮。
    assert app.memory.save_turn.call_args.kwargs["summary"]["through_turn_id"] == 19  # 游标准确前移到新边界。


def test_prompt_order_and_new_message_usage_exclude_summary_and_profile(summary_chat, monkeypatch):  # 资料不进入本轮输出与费用。
    app, graph, outputs = summary_chat  # 使用已有摘要和 8 轮原文。
    app.memory.profile.return_value = {"姓名": {"value": "小舟", "forgotten": False, "expires_on": None}}  # 当前有效画像。
    app.chat("summary-study", "当前问题")  # 未超过阈值，不重新生成摘要。
    messages = graph.invoke.call_args.args[0]["messages"]  # 检查模型实际收到的消息顺序。
    assert "更早的摘要" in messages[0].text and isinstance(messages[0], HumanMessage)  # 摘要是历史资料消息。
    assert "小舟" in messages[1].text and "共享结构化记忆" in messages[1].text  # 画像紧跟摘要。
    assert [message.text for message in messages[2:6]] == ["原问题0", "原回答0", "原问题1", "原回答1"]  # 原文按完整问答排列。
    assert "当前问题" in messages[-1].text and "本轮检索资料" in messages[-1].text  # 当前问题与 RAG 位于最后。
    assert agent_module.create_agent.call_args.kwargs["system_prompt"] == SYSTEM_PROMPT  # 固定规则里不再混入画像。
    assert app.memory.save_turn.call_args.kwargs["usage"]["current"]["total_tokens"] == 112  # 旧摘要、画像和历史不重复计费。


def test_profile_follows_fixed_rules_when_summary_is_absent(summary_chat):  # 首次摘要之前依然有明确画像位置。
    app, graph, outputs = summary_chat  # 使用同一套真实主流程。
    app.memory.context.return_value = ([], {}, {})  # 当前会话尚无历史与摘要。
    app.chat("empty-study", "新问题")  # 正常完成一轮，不添加空摘要消息。
    messages = graph.invoke.call_args.args[0]["messages"]  # 查看输入资料。
    assert len(messages) == 2 and "共享结构化记忆" in messages[0].text  # 仅有画像与本轮问题。
    assert app.memory.save_turn.call_args.kwargs["summary"] is None  # 未摘要状态保存 SQL NULL。


def test_summary_and_history_remain_stable_when_profile_or_rag_changes(summary_chat):  # 缓存排列优化只稳定允许复用的材料。
    app, graph, outputs = summary_chat  # 两轮都不达到摘要阈值。
    app.chat("summary-study", "第一轮问题")  # 保存第一轮实际输入。
    first = graph.invoke.call_args.args[0]["messages"]  # 消息列表每轮重新构造，不会原地覆盖。
    app.memory.profile.return_value = {"偏好": {"value": "简明说明", "forgotten": False, "expires_on": None}}  # 第二轮画像发生变化。
    app.knowledge.context.return_value = "[new.md#0]\n新检索结果"  # 第二轮检索资料也发生变化。
    app.chat("summary-study", "第二轮问题")  # 重新组织动态资料。
    second = graph.invoke.call_args.args[0]["messages"]  # 查看第二轮真正输入。
    assert first[0].text == second[0].text  # 未重压缩时累计摘要原样复用。
    assert first[1].text != second[1].text  # 画像紧跟摘要并反映最新状态。
    assert first[2:-1] == second[2:-1]  # 未压缩历史原话与顺序保持稳定。
    assert first[-1].text != second[-1].text  # 本轮问题和 RAG 独立变化。
    app.llm.invoke.assert_not_called()  # 不为稳定前缀而每轮重新润色摘要。


def test_summary_runs_with_reflection_disabled_and_ignores_auxiliary_totals(summary_chat, monkeypatch):  # 文件共用但功能独立。
    app, graph, outputs = summary_chat  # 默认关闭答案检查。
    outputs[:] = [model_reply(50001)]  # 主问答达到摘要条件。
    app.chat("summary-study", "需要摘要")  # 不开启检查开关也要调用摘要。
    app.llm.invoke.assert_called_once()  # 这一次调用只属于摘要。
    app.settings = replace(app.settings, reflection_enabled=True)  # 第二轮开启答案检查。
    monkeypatch.setattr(reflection_module, "reflect_answer", Mock(return_value=("检查后的回答", {}, [model_reply(90000)])))  # 检查用量比阈值大。
    outputs[:] = [model_reply(100)]  # 主问答总量没有超过阈值。
    app.chat("summary-study", "不应因反思用量摘要")  # 反思是独立请求，不能触发主历史摘要。
    assert app.llm.invoke.call_count == 1  # 第二轮没有额外摘要调用。


def test_no_old_turns_means_no_empty_summary_request(summary_chat):  # 一条超长新问题也不能凭空压缩旧历史。
    app, graph, outputs = summary_chat  # 模拟新会话。
    app.memory.context.return_value = ([], {}, {})  # 没有可压缩的完整旧问答。
    outputs[:] = [model_reply(50001)]  # 主问答总量已经超过阈值。
    app.chat("empty-study", "超长新问题")  # 保留本轮原文并正常保存。
    app.llm.invoke.assert_not_called()  # 不向模型发送空摘要任务。


def test_few_expensive_turns_can_compress_while_retaining_current_turn(summary_chat):  # 未满 6 轮也可能已经超过五万 token。
    app, graph, outputs = summary_chat  # 使用默认保留 6 轮的配置。
    turns, previous, state = app.memory.context.return_value  # 保存两条真实 ID 的原文作为超长场景。
    app.memory.context.return_value = (turns[:2], previous, state)  # 旧问答只有 2 轮，但每轮可能很长。
    outputs[:] = [model_reply(50001)]  # 已经实际超过阈值，不能因轮数少而完全不压缩。
    app.chat("summary-study", "本轮问题")  # 生成摘要并仅保留本轮原文。
    payload = json.loads(app.llm.invoke.call_args.args[0][1].text)  # 检查没有切断半轮或纳入未保存本轮。
    assert payload["turns"] == turns[:2]  # 两个完整旧问答全部纳入摘要。
    assert app.memory.save_turn.call_args.kwargs["summary"]["through_turn_id"] == 13  # 边界仍使用最后一条旧记录真实 ID。


def test_missing_usage_does_not_become_zero_or_trigger_summary(summary_chat):  # 缺失用量需要准确反馈。
    app, graph, outputs = summary_chat  # 其余流程保持正常。
    outputs[:] = [AIMessage(content="未提供用量的回答")]  # 模拟接口没有 usage。
    result = app.chat("summary-study", "当前问题")  # 仍可保存有效回答。
    assert "未提供完整 token" in result["memory_warning"]  # 明确没有完成阈值判断。
    assert result["usage"]["current"]["total_tokens"] is None  # 不把不完整消耗显示成零。
    app.llm.invoke.assert_not_called()  # 不凭空判定需要摘要。


def test_total_can_use_complete_returned_input_and_output(summary_chat):  # 总量字段缺失时无需估算。
    app, graph, outputs = summary_chat  # 保留真实消息对象与其余流程。
    message = model_reply(50001)  # 输入和输出是完整返回值。
    message.usage_metadata["total_tokens"] = None  # 模拟提供两项分量却没有总量的服务。
    outputs[:] = [message]  # 该消息仍是最后一次主问答返回。
    app.chat("summary-study", "当前问题")  # 以完整输入加输出判断。
    app.llm.invoke.assert_called_once()  # 50001 大于阈值，摘要正常触发。


@pytest.mark.parametrize("failure", [RuntimeError("不要泄露的请求详情"), model_reply(110, "", response_metadata={}), model_reply(110, "残缺摘要", response_metadata={"finish_reason": "length"}), model_reply(110, "错误工具请求", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "wrong-summary-tool"}])])  # 覆盖服务失败、空结果、服务端截断与不允许的工具请求。
def test_failed_summary_preserves_snapshot_and_known_usage(summary_chat, failure):  # 失败时不能推进游标。
    app, graph, outputs = summary_chat  # 原快照指向 ID 4。
    old_summary = app.memory.context.return_value[2]  # 保存失败前的状态。
    outputs[:] = [model_reply(50001)]  # 回答已成功，摘要阶段才失败。
    if isinstance(failure, Exception):  # 请求本身失败时没有模型消息。
        app.llm.invoke.side_effect = failure  # 仅模拟摘要请求报错。
    else:  # 空或截断消息仍有真实用量。
        app.llm.invoke.return_value = failure  # 函数需要校验并保留原始返回。
    result = app.chat("summary-study", "当前问题")  # 问答保存应继续执行。
    assert app.memory.save_turn.call_args.kwargs["summary"] == old_summary  # 摘要正文和覆盖位置一起保持原样。
    assert "历史摘要失败" in result["memory_warning"] and "不要泄露" not in result["memory_warning"]  # 页面显示准确且简短的警告。
    expected = None if isinstance(failure, Exception) else 50001 + 12 + 110  # 无返回为未知，已返回的消耗仍累计。
    assert result["usage"]["current"]["total_tokens"] == expected  # 校验失败不等于请求未消耗。


def test_summary_has_no_application_length_limit(summary_chat):  # 验证没有偷偷保留 1000 token 或字符上限。
    app, graph, outputs = summary_chat  # 主问答触发摘要。
    outputs[:] = [model_reply(50001)]  # 单次返回总量超过阈值。
    long_summary = "必要的历史背景。" * 2000  # 超过旧方案建议的摘要长度。
    app.llm.invoke.return_value = model_reply(110, long_summary)  # 模型返回长但完整的摘要。
    app.chat("summary-study", "当前问题")  # 保存函数校验后的全文。
    assert app.memory.save_turn.call_args.kwargs["summary"]["text"] == long_summary  # 应用不截断摘要。
    assert app.llm.invoke.call_args.kwargs == {"extra_body": {"enable_thinking": False}}  # 没有 max_tokens 等输出限制。


def test_profile_merge_fallback_preserves_summary_and_identical_usage(summary_chat):  # 备用保存路径不能漏摘要或重复累计。
    app, graph, outputs = summary_chat  # 摘要正常生成。
    outputs[:] = [model_reply(50001)]  # 触发压缩。
    app.memory.save_turn.side_effect = [ValueError("画像容量超限"), None]  # 第一次事务回滚，第二次只存问答。
    result = app.chat("summary-study", "当前问题")  # 完整执行降级路径。
    first, second = app.memory.save_turn.call_args_list  # 比较两次保存参数。
    assert first.kwargs["summary"] == second.kwargs["summary"] == {"text": "新的累计摘要", "through_turn_id": 16}  # 摘要和游标均保留。
    assert first.kwargs["usage"] == second.kwargs["usage"] == result["usage"]  # 同一份统计不能再次相加。
    assert result["usage"]["current"]["total_tokens"] == 50123  # 主问答、提取与摘要各计一次。


def test_summary_materials_do_not_override_latest_profile_or_extract_old_facts(summary_chat):  # 摘要仍是较低可信的历史资料。
    app, graph, outputs = summary_chat  # 使用真实提示词构造逻辑。
    app.memory.profile.return_value = {"姓名": {"value": None, "forgotten": True, "expires_on": None}}  # 用户已经遗忘姓名。
    app.chat("summary-study", "更正：学习目标是 Python")  # 本轮只提取新原话。
    messages = graph.invoke.call_args.args[0]["messages"]  # 摘要与画像都有资料边界。
    profile = json.loads(messages[1].text.splitlines()[-1])  # 画像紧跟摘要。
    assert "value" not in profile["姓名"] and profile["姓名"]["status"] == "已遗忘"  # 失效旧值不能回到画像上下文。
    assert "当前画像优先于冲突摘要" in SYSTEM_PROMPT  # 主回答明确遵循最新画像规则。
    assert agent_module.extract_changes.call_args.args[1] == "更正：学习目标是 Python"  # 摘要不作为新个人事实来源。


@pytest.mark.parametrize("updates", [{"context_token_budget": 0}, {"summary_keep_turns": 0}])  # 新配置需要正数。
def test_summary_settings_reject_invalid_values(settings, updates):  # 错误应在连接任何服务之前发现。
    with pytest.raises(ValueError, match="摘要阈值"):  # 检查用户能理解的错误提示。
        replace(settings, **updates)  # dataclass 创建时自动调用配置校验。


def test_new_summary_environment_settings_replace_old_window(monkeypatch, settings):  # 本机保留旧变量时也不能恢复固定窗口。
    monkeypatch.setattr(config_module, "load_dotenv", Mock())  # 不读取真实密钥文件。
    env = {"LLM_API_KEY": settings.llm_api_key, "LLM_BASE_URL": settings.llm_base_url, "MYSQL_USER": settings.mysql_user, "MYSQL_PASSWORD": settings.mysql_password, "CONTEXT_TOKEN_BUDGET": "65000", "SUMMARY_KEEP_TURNS": "2", "MEMORY_TURNS": "1", "MEMORY_CHAR_BUDGET": "1"}  # 所有值均为虚构测试配置。
    monkeypatch.setattr(config_module, "os", SimpleNamespace(environ=env))  # 只替换配置模块的环境视图。
    loaded = config_module.load_settings()  # 执行真实类型转换和配置校验。
    assert loaded.context_token_budget == 65000 and loaded.summary_keep_turns == 2  # 新变量转换为整数。
    assert not hasattr(loaded, "memory_turns") and not hasattr(loaded, "memory_char_budget")  # 旧窗口配置已不构成运行逻辑。


def test_empty_main_output_does_not_extract_summarize_or_save(summary_chat):  # 新增输入资料不能被误当成最终回答。
    app, graph, outputs = summary_chat  # 图正常回传输入，但没有新增任何模型消息。
    outputs.clear()  # 模拟主模型没有有效返回。
    with pytest.raises(RuntimeError, match="有效的最终回答"):  # 业务错误需要明确，不暴露列表下标异常。
        app.chat("summary-study", "当前问题")  # 主问答在验收阶段中止。
    agent_module.extract_changes.assert_not_called()  # 没有回答就不能提取个人事实。
    app.llm.invoke.assert_not_called()  # 也不能开始历史摘要。
    app.memory.save_turn.assert_not_called()  # 失败轮次不保存成成功问答。
