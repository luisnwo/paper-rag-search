from pathlib import Path
import pickle
import re
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 retrieval."""
    return re.findall(r"\b\w+\b", text.lower())

def build_bm25_index(
    collection,
    output_path: Path,
) -> None:
    """Build a BM25 index from a ChromaDB collection.

    Args:
        collection: ChromaDB collection containing indexed chunks.
        output_path: File in which the BM25 index is stored.
    """
    data = collection.get(
        include=["documents", "metadatas"],
    )

    documents = data["documents"]
    metadatas = data["metadatas"]
    ids = data["ids"]

    tokenized_docs = [_tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_docs)

    payload = {
        "bm25": bm25,
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def load_bm25_index(
    path: Path,
):
    """Load a BM25 index."""
    with path.open("rb") as f:
        return pickle.load(f)


def bm25_search(
    bm25_data,
    query: str,
    top_k: int = 20,
) -> list[dict]:
    """Retrieve chunks using BM25 lexical search.

    Args:
        bm25_data: Data returned by ``load_bm25_index``.
        query: Search query.
        top_k: Maximum number of chunks to return.

    Returns:
        Retrieved chunks with BM25 scores and metadata.
    """
    bm25 = bm25_data["bm25"]
    documents = bm25_data["documents"]
    metadatas = bm25_data["metadatas"]
    ids = bm25_data["ids"]

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for rank, i in enumerate(ranked_indices, start=1):
        results.append(
            {
                "id": ids[i],
                "source": metadatas[i]["source"],
                "chunk_index": metadatas[i]["chunk_index"],
                "text": documents[i],
                "score": float(scores[i]),
                "rank": rank,
            }
        )

    return results

def rerank_chunks(
    query: str,
    candidates: list[dict],
    model: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:
    """Rerank retrieved chunks using a cross-encoder.

    Args:
        query: Search query.
        candidates: Candidate chunks from initial retrieval.
        model: Cross-encoder reranking model.
        top_k: Number of chunks to return after reranking.

    Returns:
        The highest-scoring chunks, ordered by reranker score.
    """
    if not candidates:
        return []

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    results = []

    for rank, (candidate, score) in enumerate(
        ranked[:top_k],
        start=1,
    ):
        result = dict(candidate)
        result["rerank_score"] = float(score)
        result["rank"] = rank
        results.append(result)

    return results

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge ranked retrieval results using Reciprocal Rank Fusion.

    Args:
        result_lists: Multiple ranked result lists.
        k: RRF smoothing constant.

    Returns:
        A single ranked list of unique chunks.
    """
    fused = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )