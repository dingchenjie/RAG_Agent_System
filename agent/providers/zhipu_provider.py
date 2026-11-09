"""
智谱 AI 的具体实现
"""
import json
from zhipuai import ZhipuAI
from .base import BaseLLMProvider, LLMResponse, ToolCall
from typing import List, Dict

class ZhipuProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1
        )
        msg = response.choices[0].message

        # 将智谱的 tool_calls 转换为标准的 ToolCall 对象
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LLMResponse(content=msg.content, tool_calls=tool_calls, raw=response)

    def format_assistant_message(self, response: LLMResponse) -> Dict:
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                    }
                } for tc in response.tool_calls
            ]
        }

    def format_tool_result(self, tool_call_id: str, result: str) -> Dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }