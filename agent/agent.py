"""
ReAct Agent 主循环
推理 → 调用工具 → 观察结果 → 继续推理 → 最终回答
"""
import json
from typing import List, Dict, Any
from zhipuai import ZhipuAI
from .tools import TOOLS_SCHEMA, get_tool_executor
from core import config

MAX_ITERATIONS = 8  # 最大迭代次数，防止死循环


class Agent:
    def __init__(self, collection):
        self.client = ZhipuAI(api_key=config.ZHIPU_API_KEY)
        self.collection = collection
        self.messages: List[Dict[str, Any]] = []
        self.tool_calls_log: List[Dict] = []  # 记录每步调用，用于评估单步成功率

    def _call_llm(self):
        """调用大模型，返回响应对象"""
        return self.client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=self.messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1
        )

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具并返回结果"""
        executor = get_tool_executor(tool_name, self.collection)
        # 根据参数名动态调用
        if tool_name == "search_paper":
            return executor(arguments.get("question", ""))
        elif tool_name == "calculate":
            return executor(arguments.get("expression", ""))
        elif tool_name == "get_current_time":
            return executor()
        else:
            return f"错误：未知工具 {tool_name}"

    def run(self, user_input: str) -> str:
        """
        运行 Agent，处理用户输入，返回最终回答
        """
        # 1. 初始化对话
        self.messages = [
            {
                "role": "system",
                "content": "你是一个学术论文助手 Agent。你可以使用工具来辅助回答问题。"
                           "当需要论文信息时调用 search_paper，需要计算时调用 calculate，"
                           "需要时间时调用 get_current_time。"
            },
            {"role": "user", "content": user_input}
        ]

        # 2. ReAct 循环
        for iteration in range(MAX_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1} ---")

            response = self._call_llm()
            message = response.choices[0].message

            # 3. 如果模型直接输出文本（没有工具调用），返回答案
            if not message.tool_calls:
                return message.content

            # 4. 处理工具调用
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"模型决策：调用工具 {tool_name}，参数 {arguments}")

                # 执行工具
                try:
                    result = self._execute_tool(tool_name, arguments)
                    tool_status = "success"
                except Exception as e:
                    result = f"工具执行失败：{e}"
                    tool_status = "failed"

                # 记录日志（用于单步成功率统计）
                self.tool_calls_log.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "status": tool_status
                })

                print(f"工具结果：{result[:80]}...")

                # 5. 把工具结果拼回上下文
                # 注意：智谱 API 返回的 message 是 Pydantic 对象，需要转成字典
                assistant_message = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                }
                self.messages.append(assistant_message)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # 6. 达到最大迭代次数仍未完成
        return "抱歉，处理时间过长，请简化您的问题后再试。"


def create_agent(collection) -> Agent:
    """工厂函数：创建 Agent 实例"""
    return Agent(collection)