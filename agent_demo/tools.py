"""注册三个工具：模型给出工具名和参数，LangChain 执行 Python 函数。

build_tools() 用闭包保存知识库，模型只填写工具业务参数。
"""

import ast
import math
from datetime import datetime

from langchain_core.tools import BaseTool, tool

from agent_demo.rag import KnowledgeBase


def build_tools(knowledge: KnowledgeBase) -> list[BaseTool]:
    """参数：knowledge 供 knowledge_search 查询；返回三个工具对象。

    @tool 函数的三引号 docstring 是发给模型的工具说明书，不是普通注释。
    """
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式，支持数字、括号与 + - * / // %，不支持幂运算。

        参数：expression（str）是待计算的算式。返回：str，结果或失败原因。
        """
        try:
            # mode="eval" 只接受一条表达式；语法树随后再过白名单。
            tree = ast.parse(expression, mode="eval")
            allowed_nodes = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                ast.UAdd, ast.USub,
            )
            for node in ast.walk(tree):
                if type(node) not in allowed_nodes:
                    raise ValueError("仅支持数字、括号、+ - * / // %")
                # bool 是 int 的子类，用精确类型比较挡掉 True+1。
                if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
                    raise ValueError("仅支持数字、括号、+ - * / // %")

            # 白名单已排除调用、属性访问等节点；清空 builtins 作为第二道保险。
            result = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
            if not math.isfinite(result) or abs(result) > 1e12:
                raise ValueError("结果超出教学计算器范围")
            return f"计算结果：{result:.12g}"
        except (ValueError, SyntaxError, ArithmeticError) as error:
            # 工具失败也作为普通字符串返回，模型能据此换一种算法再试。
            return f"计算失败：{error}"

    @tool
    def current_time() -> str:
        """查询电脑的当前本地时间和 UTC 偏移。

        返回：str，包含日期、时间和 UTC 偏移。
        """
        # 使用运行电脑的本地时区，不额外配置。
        now = datetime.now().astimezone()
        return f"当前时间：{now:%Y-%m-%d %H:%M:%S %z}"

    @tool
    def knowledge_search(query: str) -> str:
        """检索知识库；本轮已自动检索，资料不足时可换关键词补充。

        参数：query（str）是检索词。返回：str，带来源的原文或未命中提示。
        """
        # 闭包：工具记住了外层传入的知识库，模型只填写检索词。
        return knowledge.context(query)

    return [calculator, current_time, knowledge_search]
