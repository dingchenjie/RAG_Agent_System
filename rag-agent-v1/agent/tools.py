"""
Agent 工具集
每个工具包含：名称、描述、参数 schema、执行函数
"""
import json
import time
from typing import Callable, Dict, Any


# ---------- 工具执行函数 ----------

def search_paper(question: str, collection) -> str:
    """调用已有的 RAG 系统检索论文内容"""
    from core.retriever import search_and_rerank
    from core.llm_service import ask_llm

    docs = search_and_rerank(question, collection)
    answer = ask_llm(question, docs)
    return answer


def calculate(expression: str) -> str:
    """安全地执行数学计算"""
    try:
        # 只允许数字、运算符、括号、小数点
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


def get_current_time() -> str:
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ---------- 工具描述（JSON Schema 格式，发给模型） ----------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "检索 LoRA 论文内容。当用户询问论文相关问题时调用，如'LoRA的原理是什么'、'论文里秩r设为多少'等。如果用户的问题和论文无关，不要调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户关于论文的问题，用英文提问检索效果更好"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算。当用户需要进行数值运算时调用，如'计算 8 除以 8'、'根号16是多少'等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '8/8'、'3*4+2'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当用户询问'现在几点了'、'今天日期'时调用。",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# ---------- 工具名到执行函数的映射 ----------

def get_tool_executor(tool_name: str, collection) -> Callable:
    """根据工具名返回对应的执行函数"""
    if tool_name == "search_paper":
        return lambda question: search_paper(question, collection)
    elif tool_name == "calculate":
        return lambda expression: calculate(expression)
    elif tool_name == "get_current_time":
        return lambda: get_current_time()
    else:
        raise ValueError(f"未知工具: {tool_name}")