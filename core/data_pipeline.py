# 数据清洗与入库逻辑
import os
os.environ['HF_HUB_OFFLINE'] = '1'

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
from zhipuai import ZhipuAI
from . import config


def load_and_split_pdf(file_path: str):
    print(f"正在加载 PDF: {file_path}")
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = splitter.split_text(text)
    print(f"切分完成，共 {len(chunks)} 个文本块")
    return chunks


def filter_redundant_chunks(chunks: list):
    print(f"正在进行冗余样本筛选 (阈值: {config.SIMILARITY_THRESHOLD})...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = model.encode(chunks, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    kept_indices = []
    for i in range(len(chunks)):
        if not any(sim_matrix[i][kept_idx] > config.SIMILARITY_THRESHOLD for kept_idx in kept_indices):
            kept_indices.append(i)

    filtered_chunks = [chunks[i] for i in kept_indices]
    print(f"筛选完成！剔除冗余后保留: {len(filtered_chunks)} 个")
    return filtered_chunks


def build_vector_db(chunks: list):
    print("正在向量化并存入 ChromaDB...")
    client = ZhipuAI(api_key=config.ZHIPU_API_KEY)
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

    try:
        chroma_client.delete_collection("rag_knowledge")
    except:
        pass
    collection = chroma_client.create_collection(name="rag_knowledge")

    for i, chunk in enumerate(chunks):
        vec = client.embeddings.create(model=config.EMBEDDING_API_MODEL, input=chunk).data[0].embedding
        collection.add(embeddings=[vec], documents=[chunk], ids=[f"chunk_{i}"])

    print(f"向量数据库构建完成，共 {collection.count()} 条记录")
    return collection
