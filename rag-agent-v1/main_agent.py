"""
Agent 演示入口
"""
from core.data_pipeline import load_and_split_pdf, filter_redundant_chunks, build_vector_db
from agent.agent import create_agent


def main():
    # 1. 构建知识库（复用你的 RAG 核心）
    print("正在构建知识库...")
    raw_chunks = load_and_split_pdf("lora.pdf")
    clean_chunks = filter_redundant_chunks(raw_chunks)
    collection = build_vector_db(clean_chunks)
    print("知识库构建完成。\n")

    # 2. 创建 Agent
    agent = create_agent(collection)

    # 3. 交互循环
    print("========== 学术论文助手 Agent 就绪 ==========")
    print("试试这些问题：")
    print("  1. LoRA的秩r通常设置为多少？")
    print("  2. 如果r=8，alpha=8，它们的比值是多少？")
    print("  3. 现在几点了？")
    print("输入 'quit' 退出\n")

    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['quit', 'q', 'exit']:
            print("再见！")
            break
        if not question:
            continue

        answer = agent.run(question)
        print(f"\nAgent 回答：{answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()