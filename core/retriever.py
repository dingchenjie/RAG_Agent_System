# 检索、重排与翻译逻辑
import os
os.environ['HF_HUB_OFFLINE'] = '1'

from sentence_transformers import CrossEncoder
from zhipuai import ZhipuAI
from . import config

# 初始化模型 (全局加载)
print("正在加载 Rerank 模型...")
reranker_model = CrossEncoder(config.RERANK_MODEL_NAME)
zhipu_client = ZhipuAI(api_key=config.ZHIPU_API_KEY)


def translate_query_if_chinese(query: str) -> str:
    is_chinese = any('\u4e00' <= char <= '\u9fa5' for char in query)
    if is_chinese:
        prompt = f"Translate to English. Only output the translated text.\nChinese: {query}\nEnglish:"
        res = zhipu_client.chat.completions.create(
            model=config.LLM_MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        return res.choices[0].message.content.strip()
    return query


def search_and_rerank(question: str, collection):
    search_query = translate_query_if_chinese(question)

    q_vec = zhipu_client.embeddings.create(model=config.EMBEDDING_API_MODEL, input=search_query).data[0].embedding
    results = collection.query(query_embeddings=[q_vec], n_results=config.RECALL_K)
    candidate_docs = results['documents'][0]

    rerank_inputs = [[search_query, doc] for doc in candidate_docs]
    scores = reranker_model.predict(rerank_inputs)

    doc_score_pairs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    final_docs = [pair[0] for pair in doc_score_pairs[:config.RERANK_TOP_K]]

    return final_docs
