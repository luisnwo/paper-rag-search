"""
Retrieval from the ChromaDB index
"""

from pathlib import Path
from config import COLLECTION_NAME

def retrieve_chunks_dense(
    collection,
    embed_model,
    query: str,
    top_k: int = 5,
    context: int = 1,
    source: str | None = None,
) -> list[dict]:
    """ Retrieve the most relevant chunks for a query.

    The query is embedded using ``embed_model`` and used to search
    ``collection`` for the most relevant chunks. Optionally, retrieval
    can be restricted to a single source file. For each matching chunk,
    additional surrounding chunks can be included using ``context``.

    Args:
        collection: Vector database collection used for similarity search.
        embed_model: Embedding model used to convert the query into a vector.
        query: Text query used to retrieve relevant chunks.
        top_k: Maximum number of matching chunks to retrieve.
        context: Number of neighboring chunks to include before and after
            each retrieved chunk.
        source: Optional source file to restrict the search to.

    Returns:
        A list of dictionaries, one per retrieved chunk. Each dictionary
        contains the chunk's ``rank``, ``source``, ``chunk_range``,
        ``score``, and ``text``.
    """
    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    where = {"source": source} if source else None
    results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

    hits = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        score = 1 - dist  # cosine similarity
        src = meta["source"]
        idx = meta["chunk_index"]
        stem = src[:-4] if src.lower().endswith(".pdf") else src

        pieces = [(idx, doc)]
        if context > 0:
            for offset in list(range(-context, 0)) + list(range(1, context + 1)):
                neighbor_id = f"{stem}::chunk_{idx + offset}"
                try:
                    got = collection.get(ids=[neighbor_id], include=["documents"])
                except Exception:
                    continue
                if got["documents"]:
                    pieces.append((idx + offset, got["documents"][0]))
        pieces.sort(key=lambda p: p[0])
        combined = "\n\n".join(p[1].strip() for p in pieces)
        chunk_range = f"{pieces[0][0]}-{pieces[-1][0]}" if len(pieces) > 1 else str(idx)

        hits.append({"rank": rank, "source": src, "chunk_range": chunk_range, "score": score, "text": combined})

    return hits

def retrieve_chunks_hybrid(
    collection,
    embed_model,
    rerank_model,
    bm25_data,
    query: str,
    candidate_k: int = 30,
    top_k: int = 5,
    context: int = 1,
    source: str | None = None,
) -> list[dict]:
    """Retrieve chunks using dense + BM25 retrieval and reranking."""

    from reranking import bm25_search, reciprocal_rank_fusion, rerank_chunks

    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    where = {"source": source} if source else None

    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k,
        where=where,
    )

    dense_hits = []

    for doc, meta, dist, chunk_id in zip(
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0],
        dense_results["ids"][0],
    ):
        dense_hits.append(
            {
                "id": chunk_id,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "text": doc,
                "score": 1 - dist,
            }
        )

    bm25_hits = bm25_search(
        bm25_data,
        query,
        top_k=candidate_k,
    )

    if source:
        bm25_hits = [
            hit
            for hit in bm25_hits
            if hit["source"] == source
        ]

    candidates = reciprocal_rank_fusion(
        [dense_hits, bm25_hits]
    )

    # Cross-Encoder reranking
    reranked = rerank_chunks(
        query=query,
        candidates=candidates,
        model=rerank_model,
        top_k=top_k,
    )
 
    # Add surrounding context after reranking
    hits = []

    for rank, hit in enumerate(reranked, start=1):

        idx = hit["chunk_index"]
        src = hit["source"]

        stem = (
            src[:-4]
            if src.lower().endswith(".pdf")
            else src
        )

        pieces = [(idx, hit["text"])]

        if context > 0:
            for offset in (
                list(range(-context, 0))
                + list(range(1, context + 1))
            ):
                neighbor_id = (
                    f"{stem}::chunk_{idx + offset}"
                )

                try:
                    got = collection.get(
                        ids=[neighbor_id],
                        include=["documents"],
                    )
                except Exception:
                    continue

                if got["documents"]:
                    pieces.append(
                        (
                            idx + offset,
                            got["documents"][0],
                        )
                    )

        pieces.sort(key=lambda p: p[0])

        combined = "\n\n".join(
            text.strip()
            for _, text in pieces
        )

        chunk_range = (
            f"{pieces[0][0]}-{pieces[-1][0]}"
            if len(pieces) > 1
            else str(idx)
        )

        hits.append(
            {
                "rank": rank,
                "source": src,
                "chunk_range": chunk_range,
                "score": hit["rerank_score"],
                "text": combined,
            }
        )

    return hits

def query_index(
    query: str,
    db_dir: Path,
    embed_model_name: str,
    rerank_model_name: str | None = None,
    top_k: int = 5,
    candidate_k: int = 30,
    context: int = 1,
    max_chars: int = 1200,
    source: str | None = None,
) -> list[dict]:
    """ Retrieve and print the chunks most relevant to a query.

    Searches the indexed documents using the specified embedding model, 
    prints the top matching chunks, and returns them. Use
    ``generate.generate_answer`` instead when a synthesized answer
    generated by an LLM is desired.

    Args:
        query: Text query used to search the index.
        db_dir: Directory containing the vector database.
        embed_model_name: Identifier of the embedding model used for retrieval.
        top_k: Maximum number of matching chunks to display.
        context: Number of neighboring chunks to include on each side of a
            matched chunk. For example, ``context=1`` includes the chunk
            immediately before and after the matched chunk from the same
            source. Set to ``0`` to disable surrounding context.
        max_chars: Maximum number of characters of text to display per result.
            This only affects printed output, not the returned results.
        source: Optional source file used to restrict retrieval to a single
            document, for example ``"ID-123.pdf"``. Use ``list_sources`` to
            see the exact filenames available.

    Returns:
        A list of dictionaries containing the retrieved chunks. Each
        dictionary contains ``rank``, ``source``, ``chunk_range``, ``score``,
        and ``text``.

    """
    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder

    model = SentenceTransformer(embed_model_name)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(COLLECTION_NAME)

    if rerank_model_name is not None:

        from reranking import load_bm25_index
        bm25_path = db_dir / "bm25.pkl"
        bm25_data = load_bm25_index(bm25_path)
        rerank_model = CrossEncoder(rerank_model_name)
        
        hits = retrieve_chunks_hybrid(
            collection=collection,
            embed_model=model,
            rerank_model=rerank_model,
            bm25_data=bm25_data,
            query=query,
            candidate_k=candidate_k,
            top_k=top_k,
            context=context,
            source=source,
        )
    else: 
        hits = retrieve_chunks_dense(
            collection=collection,
            embed_model=model,
            query=query,
            top_k=top_k,
            context=context,
            source=source,
        )

    scope = f' (restricted to {source})' if source else ""
    print(f'\nTop {top_k} results for: "{query}"{scope}')

    if rerank_model_name is not None:
        print("(score = cross-encoder reranker score; higher = more relevant to the query)\n")
    else:
        print("(score = cosine similarity; higher = more semantically similar to the query)\n")

    for hit in hits:
        print(f"{'=' * 70}")
        print(f"#{hit['rank']}  {hit['source']}  (chunk {hit['chunk_range']})  score={hit['score']:.3f}")
        print(f"{'-' * 70}")
        text = hit["text"][:max_chars]
        if len(hit["text"]) > max_chars:
            text += " [...]"
        print(text)
        print()

    return hits


def list_sources(db_dir: Path) -> None:
    """Print every distinct source filename currently in the collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(COLLECTION_NAME)
    existing = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in existing["metadatas"] if m and "source" in m})

    if not sources:
        print("No sources found in this collection.")
        return

    print(f"{len(sources)} source file(s) in the index:\n")
    for s in sources:
        print(f"  {s}")
