from core.data_pipeline import load_and_split_pdf, filter_redundant_chunks, build_vector_db
from core.retriever import search_and_rerank
from core.llm_service import ask_llm


def main():
    raw_chunks = load_and_split_pdf("lora.pdf")
    clean_chunks = filter_redundant_chunks(raw_chunks)
    collection = build_vector_db(clean_chunks)

    print("\n========== 知识库问答系统就绪 ==========")
    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['quit', 'q']: break
        if not question: continue

        docs = search_and_rerank(question, collection)
        answer = ask_llm(question, docs)
        print(f"\n回答：{answer}\n" + "-" * 50)


if __name__ == "__main__":
    main()
