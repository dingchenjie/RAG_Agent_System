"""
RAG 系统 Faithfulness（忠实度）量化评估脚本
使用 RAGAS 框架，调用智谱 API 进行评估
"""
import os
import sys
import time
import json
from typing import List, Tuple

# 解决 ragas 导入时的 VertexAI 缺失问题
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

# 添加项目根目录到路径，方便导入 core 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_pipeline import load_and_split_pdf, filter_redundant_chunks, build_vector_db
from core.retriever import search_and_rerank
from core.llm_service import ask_llm
from core import config

from ragas import SingleTurnSample
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatZhipuAI

# ==================== 配置区 ====================
PDF_PATH = "lora.pdf"                      # PDF 文件路径
EVAL_DATASET_PATH = "eval_dataset.json"    # 评测集 JSON 文件
# ================================================

def load_eval_dataset(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_rag_and_collect(question: str, collection) -> Tuple[str, List[str]]:
    """运行一次 RAG 流程，返回回答和上下文"""
    docs = search_and_rerank(question, collection)
    answer = ask_llm(question, docs)
    return answer, docs

def main():
    # 1. 初始化智谱评估专用 LLM
    zhipu_eval_llm = ChatZhipuAI(
        model=config.LLM_MODEL_NAME,
        api_key=config.ZHIPU_API_KEY,
        temperature=0.0,
        max_retries=3,
        request_timeout=60,
    )
    # 替换 faithfulness 默认的 OpenAI 后端
    faithfulness.llm = LangchainLLMWrapper(zhipu_eval_llm)

    # 2. 构建知识库（与主流程一致）
    print("正在构建知识库...")
    raw_chunks = load_and_split_pdf(PDF_PATH)
    clean_chunks = filter_redundant_chunks(raw_chunks)
    collection = build_vector_db(clean_chunks)
    print("知识库构建完成。\n")

    # 3. 加载评测数据集
    eval_items = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"正在对 {len(eval_items)} 个问题进行评估...")

    questions, answers, contexts_list = [], [], []

    for idx, item in enumerate(eval_items):
        q = item["question"]
        ans, ctxs = run_rag_and_collect(q, collection)
        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctxs)
        print(f"  Q{idx+1}: {q}")
        print(f"  A: {ans[:80]}...")
        print("---")
        time.sleep(2)  # 降低 API 调用频率

    # 4. 逐条串行计算 Faithfulness
    print("\n正在逐条计算 Faithfulness（串行评估，较慢但稳定）...")
    faithfulness_scores = []

    for idx, (q, a, ctxs) in enumerate(zip(questions, answers, contexts_list)):
        sample = SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctxs
        )
        success = False
        for retry in range(5):
            try:
                score = faithfulness.single_turn_score(sample)
                faithfulness_scores.append(score)
                print(f"  Q{idx+1}: {score:.4f}")
                success = True
                break
            except Exception as e:
                print(f"  Q{idx+1} 第{retry+1}次评估失败: {e}")
                if retry < 4:
                    time.sleep(3 * (retry + 1))
        if not success:
            print(f"  Q{idx+1}: 最终失败，标记为 0")
            faithfulness_scores.append(0.0)
        time.sleep(2)

    # 5. 输出结果
    print("\n========== 评估结果 ==========")
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    print(f"Faithfulness (忠实度) 平均分: {avg_faithfulness:.4f}")
    print("\n各样本得分：")
    for idx, score in enumerate(faithfulness_scores):
        print(f"  Q{idx+1}: {score:.4f}")

    # 6. 保存结果到 CSV
    import pandas as pd
    result_df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "faithfulness": faithfulness_scores
    })
    result_df.to_csv("faithfulness_result.csv", index=False, encoding="utf-8")
    print("\n详细结果已保存至 faithfulness_result.csv")

if __name__ == "__main__":
    main()