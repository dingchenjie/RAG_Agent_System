import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.data_pipeline import load_and_split_pdf, filter_redundant_chunks, build_vector_db
from core.retriever import search_and_rerank
from core.llm_service import ask_llm

app = FastAPI(title="Advanced RAG API")

# 启动时构建知识库
print("🚀 正在初始化知识库...")
raw_chunks = load_and_split_pdf("lora.pdf")
clean_chunks = filter_redundant_chunks(raw_chunks)
collection = build_vector_db(clean_chunks)

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        docs = search_and_rerank(request.question, collection)
        answer = ask_llm(request.question, docs)
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
