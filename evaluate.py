"""真实效果评测：使用配置的百炼 API、MySQL 和 Chroma，无运行模式开关。"""

# tests/ 用假对象测零件，免费、秒级；本文件真调模型、真写数据库，测整机是否可用。
# 两条链路分开：想测假的去 pytest，想测真的运行本文件。

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

logger = logging.getLogger("agent_demo.evaluate")


def run_evaluation(app: AgentApp, run_id: str) -> dict[str, Any]:
    """参数：独立评测应用、运行 ID；返回通过数、总数和逐题结果。"""
    results = []

    def ask(name: str, session: str, question: str, words: list[str],
            expected_tool: str | None = None) -> dict[str, Any] | None:
        """运行一道评测题并追加结果；异常时记录失败并继续下一题。"""
        try:
            result = app.chat(session, question)
            # 关键词只做字面包含；同义表达可能误判，报告保留原文供人工复核。
            answer_ok = all(word in result["answer"] for word in words)
            used_tools = [item["tool"] for item in result["trace"] if item.get("kind") != "reflection"]
            tool_ok = expected_tool is None or expected_tool in used_tools
            results.append({
                "case": name,
                "passed": answer_ok and tool_ok,
                "answer": result["answer"], "trace": result["trace"],
                "memory_warning": result["memory_warning"],
            })
            return result
        except Exception as error:
            logger.exception("评测题失败：%s", name)
            # 只记异常类型，不记可能带请求内容的详情。
            results.append({"case": name, "passed": False, "error": type(error).__name__})
            return None

    prefix = "eval_" + run_id

    # ---- 基本功：答案和工具都要对 ----
    ask("计算器", prefix + "_math", "请使用计算器计算 (12+8)*3。", ["60"], "calculator")
    ask("当前时间", prefix + "_time", "请调用时间工具，并写出当前完整日期和时间。",
        [str(datetime.now().year)], "current_time")
    # 问题里没有“请查知识库”，用于验证每轮自动检索。
    ask("无检索提示词的知识问答", prefix + "_rag",
        "星舟课程退款需同时满足哪些条件？保留 7 天和 20% 的数字写法并引用文件名。",
        ["7", "20%", "course.md#"], "knowledge_search")

    # ---- 记忆：短期原文、真实摘要、摘要后的接续 ----
    memory_session = prefix + "_memory"

    # 准备动作，不计入评测题数。
    app.chat(memory_session, "我叫小明，请记住我的名字。练习示例项目代号是星河练习，示例不属于我的个人资料。")

    ask("短期记忆", memory_session, "我叫什么名字？", ["小明"])

    normal_settings = app.settings
    try:
        # 临时调低阈值，通过真实回答触发摘要，避免制造五万 token 的付费输入。
        app.settings = replace(normal_settings, context_token_budget=1, summary_keep_turns=1)
        ask("真实回答后触发摘要", memory_session, "请继续学习，并保留已有练习背景。", [])
    finally:
        app.settings = normal_settings

    remaining, _, summary = app.memory.context(memory_session)
    results.append({
        "case": "摘要持久化与覆盖边界",
        "passed": bool(summary.get("text")) and "星河练习" in summary["text"]
                  and len(remaining) == 1 and remaining[0]["id"] > summary["through_turn_id"],
    })
    ask("摘要后的历史背景接续", memory_session, "之前练习示例的项目代号是什么？", ["星河练习"])

    # ---- 换新连接直接查库，验证数据真的落盘 ----
    recovered = ConversationMemory(app.settings)
    other_user = ConversationMemory(replace(app.settings, user_id=prefix + "_other"))

    try:
        saved_turns = recovered.recent(memory_session)
        restored = any("小明" in row["assistant_text"] for row in saved_turns)
        isolated = not recovered.recent(prefix + "_other")
        results.append({"case": "MySQL 重连与会话隔离", "passed": restored and isolated})

        profile = recovered.profile()
        name_keys = [key for key, fact in profile.items() if fact["value"] == "小明" and not fact["forgotten"]]
        # bool(name_keys) 防“空洞通过”：提取功能彻底坏掉时，后面几个 not 条件也会成立。
        results.append({"case": "结构化记忆持久化与用户隔离", "passed": bool(name_keys)
                        and not other_user.profile() and not other_user.recent(memory_session)
                        and not other_user.list_sessions()})
    finally:
        recovered.close()
        other_user.close()

    # ---- 更正与遗忘 ----
    new_session = prefix + "_new"
    ask("新会话共享个人记忆", new_session, "我叫什么名字？", ["小明"])

    # 在 new_session 更正，回旧会话验证，证明画像跨会话生效。
    app.chat(new_session, "更正：我现在叫小舟，请更新我的姓名。")
    ask("旧会话读取更正后的记忆", memory_session, "我现在叫什么名字？", ["小舟"])

    app.chat(new_session, "请忘记我的姓名，别再把它作为当前个人资料。")
    profile = app.memory.profile()
    results.append({"case": "结构化记忆弃用标记", "passed": bool(name_keys)
                    and all(profile.get(key, {}).get("forgotten") for key in name_keys)})

    # ---- 上传临时文档再考它 ----
    name = "evaluation_notes.txt"
    app.knowledge.upload([(name, "极光计划接头口令是蓝色松果。".encode("utf-8"))])
    # 这些词在模型知识和网上都不存在，命中即可证明读的是刚上传的文档。
    ask("无检索提示词的上传问答", prefix + "_upload",
        "极光计划接头口令是什么？请引用文件名。",
        ["蓝色松果", "evaluation_notes.txt#"], "knowledge_search")

    # 开启反思时追加两题：错稿要改对，对稿不能被乱改。
    if app.settings.reflection_enabled:
        from agent_demo.reflection import reflect_answer

        materials = {
            "question": "计算 (12+8)*3。", "history": [], "profile": "{}",
            "tool_results": [{"tool": "calculator", "args": {"expression": "(12+8)*3"}, "result": "60"}],
            "answer_rules": SYSTEM_PROMPT,
        }
        for name, draft, expected_status in [
            ("反思纠正错稿", "计算结果是50。", "revised"),
            ("反思保留对稿", "已使用计算器，计算结果是60。", "passed"),
        ]:
            answer, note, _ = reflect_answer(app.llm, materials, draft)
            passed_case = note["status"] == expected_status and "60" in answer and "50" not in answer
            if expected_status == "passed":
                passed_case = passed_case and answer == draft
            results.append({"case": name, "passed": passed_case, "answer": answer, "reflection": note})

    passed = sum(row["passed"] for row in results)
    return {"passed": passed, "total": len(results), "results": results}


def main() -> None:
    """建立独立评测环境并运行用例，报告写入 data/evaluation/<运行ID>/。"""
    run_id = uuid4().hex[:12]
    folder = ROOT / "data" / "evaluation" / run_id

    # 独立用户和目录：评测聊天、画像、上传和向量库都不碰网页用户。
    settings = replace(load_settings(), data_dir=folder, user_id="eval_" + run_id)
    app = AgentApp(settings)

    try:
        report = run_evaluation(app, run_id)
    finally:
        app.memory.close()

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测结果：{report['passed']}/{report['total']} 通过")
    print(f"报告：{path}")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except Exception as error:
        logger.exception("评测运行失败")
        # 自己抛的 ValueError 给人看原话，其他异常只报类型，避免泄露请求内容。
        detail = str(error) if isinstance(error, ValueError) else "请检查模型服务、MySQL 和网络"
        raise SystemExit(f"评测失败（{type(error).__name__}）：{detail}") from None
