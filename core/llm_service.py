# 大模型生成逻辑

from zhipuai import ZhipuAI
from . import config

zhipu_client = ZhipuAI(api_key=config.ZHIPU_API_KEY)

def ask_llm(original_question: str, context_docs: list) -> str:
    context = "\n\n---\n\n".join(context_docs)
    prompt = f"""你是一个学术论文助手。请严格根据以下内容回答问题。
注意：如果用户用中文提问，请务必用中文回答。如果内容不足，请说"无法回答"。


=== 检索内容 ===
{context}
=== 结束 ===

用户问题：{original_question}
你的回答："""

    response = zhipu_client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content
