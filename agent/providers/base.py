"""
Provider 抽象基类
定义标准数据格式和统一接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ToolCall:
    """标准化的工具调用请求"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """标准化的模型响应"""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None


class BaseLLMProvider(ABC):
    """所有大模型适配器必须继承此基类"""

    @abstractmethod
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        """接收标准消息，返回标准响应"""
        pass

    @abstractmethod
    def format_assistant_message(self, response: LLMResponse) -> Dict:
        """将标准响应转为各家API需要的assistant格式"""
        pass

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, result: str) -> Dict:
        """将工具结果转为各家API需要的tool格式"""
        pass