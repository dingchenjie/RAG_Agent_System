"""向量化实现与工厂。

默认使用确定性哈希向量（零依赖、离线可跑、结果可复现），
生产可替换成语义模型（sentence-transformers / 云端 embedding）：

- ``hashing``：零依赖，测试与演示的默认路径；
- ``sentence-transformers``：本地 bge 模型，``pip install -e ".[semantic]"`` 后自动启用；
- ``openai``：走 OpenAI 兼容的 ``/embeddings`` 接口，需要 API Key 与 base_url；
- ``auto``：按"显式配置 > 本地语义模型 > 云端 Key > 哈希"选择，保证任何环境都能跑。
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import time
from typing import Protocol

import httpx

from ..errors import HermesError

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-z0-9_]+")


# 权重按模型名进程内复用：run_eval 会多次 build_state，不缓存就会反复加载语义模型
_EMBEDDER_CACHE: dict[str, object] = {}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词，中文按字 + 二元组（避免单字歧义）。"""
    lowered = text.lower()
    tokens = _WORD_RE.findall(lowered)
    cjk = _CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:], strict=False))
    return tokens


class HashingEmbedder:
    """hashed bag-of-words + 符号哈希，L2 归一化。"""

    name = "hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            idx = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAICompatEmbedder:
    """OpenAI 兼容的 ``/embeddings`` 客户端，批量 16、超时 20s、429/5xx 退避重试。"""

    name = "openai"

    def __init__(self, *, api_key: str, model: str = "embedding-3", dim: int = 0,
                 base_url: str = "", timeout_s: float = 20.0, batch_size: int = 16):
        if not api_key:
            raise HermesError("缺少 API Key（HERMES_LLM_API_KEY 或 OPENAI_API_KEY），无法使用云端 embedding",
                              code="missing_credential")
        if not base_url:
            raise HermesError("使用云端 embedding 时必须提供 base_url", code="invalid_config")
        self.api_key = api_key
        self.model = model
        # dim=0 表示"由服务端返回决定"，首次调用后回填
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload, timeout=self.timeout_s,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise HermesError(f"embedding 限流/服务错误 {response.status_code}",
                                      code="embedding_unavailable", retryable=True)
                response.raise_for_status()
                data = response.json()["data"]
                vectors = [item["embedding"] for item in data]
                for vec in vectors:
                    if self.dim and len(vec) != self.dim:
                        raise HermesError(
                            f"embedding 维度不一致：配置 {self.dim}，返回 {len(vec)}",
                            code="embedding_dim_mismatch")
                    self.dim = self.dim or len(vec)
                return vectors
            except Exception as exc:  # noqa: BLE001 统一转成带重试语义的错误
                if isinstance(exc, HermesError) and exc.code == "embedding_dim_mismatch":
                    raise  # 配置问题重试无意义
                last_error = exc
                time.sleep(0.3 * (2 ** attempt))
        raise HermesError(f"智谱 embedding 调用失败：{last_error}", code="embedding_failed")


class SentenceTransformerEmbedder:
    """本地 bge 语义向量，懒加载（首次 ``embed`` 才载入模型）。"""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", dim: int | None = None):
        self.model_name = model_name
        self.dim = dim or 512
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        cached = _EMBEDDER_CACHE.get(self.model_name)
        if cached is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise HermesError(
                    "未安装 sentence-transformers，请执行 pip install -e \".[semantic]\"",
                    code="missing_dependency") from exc
            cached = SentenceTransformer(self.model_name)
            _EMBEDDER_CACHE[self.model_name] = cached
        self._model = cached
        self.dim = int(cached.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        matrix = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in matrix]


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def build_embedder(
    kind: str = "auto",
    *,
    dim: int = 384,
    api_key: str = "",
    model: str = "",
    api_model: str = "",
    base_url: str = "",
) -> Embedder:
    """按配置构建 embedder；``auto`` 永远能返回一个可用实现，不因缺依赖而启动失败。

    ``model`` 是本地语义模型（bge），``api_model`` 是云端 provider 的模型名（如 embedding-3）——
    两者不能混用，否则会把 HuggingFace 仓库名发给云端 API 换回 400。
    """
    kind = (kind or "auto").strip().lower()
    if kind == "auto":
        if _available("sentence_transformers"):
            kind = "sentence-transformers"
        elif api_key:
            kind = "openai"
        else:
            kind = "hashing"

    if kind == "hashing":
        return HashingEmbedder(dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model or "BAAI/bge-small-zh-v1.5")
    if kind in ("openai", "zhipu"):   # zhipu 为旧配置值，等价于 openai
        return OpenAICompatEmbedder(api_key=api_key, model=api_model or "embedding-3",
                                    base_url=base_url)
    raise HermesError(f"未知的 embedding 后端：{kind}", code="invalid_config")


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False))
