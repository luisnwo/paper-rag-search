"""
Generate LLM answers from retrieved document chunks.
"""

from pathlib import Path

from config import COLLECTION_NAME, SYSTEM_PROMPT
from retrieval import query_index

def generate_answer(
    query: str,
    db_dir: Path,
    embed_model_name: str,
    top_k: int = 5,
    candidate_k: int = 30,
    context: int = 1,
    source: str | None = None,
    llm_model: str = "llama3.2",
    max_context_chars: int = 3000,
    ollama_host: str = "http://localhost:11434",
    rerank_model_name: str | None = None,

) -> str | None:
    """ Generate LLM answers from retrieved document chunks.

    Retrieves the most relevant chunks from the ChromaDB index, combines
    them into a context for the language model, and asks a local Ollama
    model to answer the query. Print answer together with the source files 
    and chunk ranges used as context.

    Args:
        query: Question to answer.
        db_dir: Directory containing the persistent ChromaDB database.
        embed_model_name: Identifier of the SentenceTransformer embedding model used to embed the query for retrieval.
        top_k: Maximum number of relevant chunks to retrieve.
        context: Number of neighboring chunks to include on each side of each retrieved chunk.
        source: Optional source PDF to restrict retrieval to a single document, for example ``"ID-123.pdf"``.
        llm_model: Name of the Ollama model used to generate the answer.
        max_context_chars: Maximum number of characters taken from each retrieved chunk when constructing the LLM context.
        ollama_host: Base URL of the local Ollama server.

    Returns:
        The generated answer as a string, or ``None`` if no matching
        chunks are found or Ollama cannot be reached successfully.
    """
    import requests

    hits = query_index(
        query=query,
        db_dir=db_dir,
        embed_model_name=embed_model_name,
        rerank_model_name=rerank_model_name,
        top_k=top_k,
        candidate_k=candidate_k,
        context=context,
        source=source,
    )

    if not hits:
        scope = f" for source '{source}'" if source else ""
        print(f"No matching chunks found{scope}.")
        return

    context_blocks = []
    for hit in hits:
        text = hit["text"][:max_context_chars]
        context_blocks.append(
            f"[Excerpt {hit['rank']} | source: {hit['source']} | chunk {hit['chunk_range']} | "
            f"similarity: {hit['score']:.3f}]\n{text}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"Excerpts:\n\n{context_str}\n\n---\n\nQuestion: {query}"

    scope = f' (restricted to {source})' if source else ""
    print(f'\nGenerating answer for: "{query}"{scope}\n')

    try:
        response = requests.post(
            f"{ollama_host}/api/chat",
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Could not reach Ollama at {ollama_host}. Is it installed and running?\n"
              f"  Install: https://ollama.com/download\n"
              f"  Then run: ollama pull {llm_model}\n"
              f"  If local server is not started automatically after install, run `ollama serve`.")
        return
    except requests.exceptions.HTTPError as e:
        print(f"Ollama returned an error ({e}). If the model isn't pulled yet, run:\n"
              f" ollama pull {llm_model}")
        return

    answer_text = response.json()["message"]["content"]

    print(f"{'=' * 70}")
    print(answer_text.strip())
    print(f"{'=' * 70}\n")
    print("Sources consulted:")
    for hit in hits:
        print(f"  - {hit['source']} (chunk {hit['chunk_range']}, similarity {hit['score']:.3f})")

    return answer_text
