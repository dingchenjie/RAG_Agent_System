"""Hermes Runtime：一个可运行、可观测、可评测的 Agent Runtime。

模块分层（与岗位 JD 的七条职责一一对应）：

- ``runtime/``      Runtime & Harness：事件循环、取消、会话、任务状态、断点恢复、调度
- ``runtime/context.py``  上下文工程：预算分配、压缩、裁剪、有效信息保留
- ``tools/``        工具与连接器：契约、注册表、版本、权限、沙箱、MCP 桥接
- ``knowledge/``    知识与 RAG：文档处理、混合检索、ACL、增量索引、引用
- ``memory/``       短期/长期记忆：提取、确认、召回、纠错、TTL、隔离
- ``agents/``       多 Agent：主 Agent 规划拆解、子 Agent 并行、共享黑板
- ``eval/``         观测与评测：轨迹、回放、版本对比、指标
"""

__version__ = "0.1.0"
