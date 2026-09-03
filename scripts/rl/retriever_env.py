"""RL 侧共用的检索器构造 —— 数据验证、训练奖励、前后评测都从这里拿。

三处必须是同一个检索器：奖励用 A 检索器算、评测用 B 检索器算的话，
训练在优化的和最后报的就不是同一个量。
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langchain_chroma import Chroma  # noqa: E402

from agentic_search import config  # noqa: E402
from agentic_search.llm import make_embeddings  # noqa: E402
from agentic_search.retrieval import (  # noqa: E402
    BM25Index,
    HybridRetriever,
    build_bm25_index,
    corpus_digest,
    load_all_chunks,
    load_bm25_index,
)

ROUTES = ("retrieve_docs", "web_search", "direct")


def make_retriever(collection: str | None = None, allowed_access=None) -> HybridRetriever:
    # 默认必须取 config.COLLECTION_NAME，不能自己写死一个名字：
    # ingest 用的是配置里那个，写死会连到一个空 collection 上。
    collection = collection or config.COLLECTION_NAME
    store = Chroma(
        collection_name=collection,
        embedding_function=make_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    chunks = load_all_chunks(store)
    if not chunks:
        raise SystemExit(
            "向量库是空的 —— 先跑 uv run python -m agentic_search.ingest"
        )
    index_path = config.CHROMA_DIR / f"bm25_{collection}.pkl"
    prebuilt: BM25Index | None = load_bm25_index(index_path, corpus_digest(chunks))
    if prebuilt is None:
        prebuilt = build_bm25_index(chunks)
    return HybridRetriever(store, chunks, allowed_access=allowed_access,
                           prebuilt=prebuilt)


def sources_of(docs) -> list[str]:
    """块的来源文件名。metadata 里键名随解析器不同，逐个兜底。"""
    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        raw = meta.get("source") or meta.get("file") or meta.get("path") or ""
        out.append(pathlib.PurePath(str(raw)).name)
    return out


def retrieval_hits(retriever, query: str, expected_source: str, k: int) -> bool:
    """query 检出的前 k 块里是否含目标文档。奖励与评测共用这一个判据。

    走 similarity_search(query, k) —— 与 tools.make_retrieve_tool 里线上那条
    路径同一个入口，奖励算的和线上跑的才是同一件事。
    """
    docs = retriever.similarity_search(query, k=k)
    return expected_source in sources_of(docs)
