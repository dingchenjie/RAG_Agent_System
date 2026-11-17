"""多轮检索的查询补全。

为什么不用 LLM 改写：改写本身要额外一次模型调用，成本和延迟都翻倍，而且同一轮
对话两次改写可能得到不同结果，评测不再可回归。这里只做确定性规则：只有"明显是
追问"的输入才把上一轮用户问题并进检索 query，其余情况原样透传。
"""
from __future__ import annotations

from ..types import Message

# 以这些词开头的输入通常指代上一轮的话题
_FOLLOW_UP_PREFIXES = (
    "那", "那么", "它", "他们", "它们", "这个", "那个", "这些", "那些",
    "上面", "刚才", "前面", "同一", "同样", "换成", "再算", "再查", "再来",
    "还有", "另外",
)
# 「上海呢？」这类省略主语的短追问
_MAX_IMPLICIT_CHARS = 12
DEFAULT_MAX_CHARS = 200

# 编排器派发的子任务 goal 会带一句指令前缀，检索前要剥掉，否则实词覆盖率被稀释
_TASK_PREFIXES = (
    "检索并总结与问题相关的资料：",
    "召回与该问题相关的用户长期记忆：",
    "完成问题中的数值计算：",
)


def is_follow_up(query: str) -> bool:
    """判断当前输入是否需要借助上一轮才能检索。"""
    text = query.strip()
    if not text:
        return False
    if text.startswith(_FOLLOW_UP_PREFIXES):
        return True
    compact = text.rstrip("？?。.!！ ")
    return len(compact) <= _MAX_IMPLICIT_CHARS and compact.endswith("呢")


def build_retrieval_query(
    current: str,
    *,
    previous: str | None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """追问时把上一轮问题并入检索 query；没有上文或不是追问则原样返回。"""
    current = current.strip()
    for prefix in _TASK_PREFIXES:
        if current.startswith(prefix):
            current = current[len(prefix):].strip()
            break
    previous = (previous or "").strip()
    if not current or not previous or not is_follow_up(current):
        return current
    merged = f"{previous} {current}"
    if len(merged) <= max_chars:
        return merged
    # 上一轮问题通常主题在前，超长时保留开头，别把话题截掉
    room = max_chars - len(current) - 1
    if room <= 0:
        return current[:max_chars]
    return f"{previous[:room]} {current}"


def previous_user_message(history: list[Message], current: str) -> str | None:
    """取最近一条内容不同于当前输入的用户消息。"""
    text = current.strip()
    for message in reversed(history):
        if message.role == "user" and message.content.strip() != text:
            return message.content
    return None
