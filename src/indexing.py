"""
Chunk, embed, and write documents to ChromaDB.
"""

import gc
from pathlib import Path
from config import COLLECTION_NAME, EMBED_MODEL_DEFAULT

def _get_or_create_collection(db_dir: Path, rebuild: bool):
    """ Opens (or creates) the ChromaDB collection with cosine similarity space.
    If ``rebuild`` is``True``, any existing collection is deleted first.

    Args:
        db_dir: Directory containing the persistent ChromaDB database.
        rebuild: Whether to delete the existing collection before indexing.

    Returns:
        The ChromaDB collection used for indexing.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(db_dir))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _already_indexed_sources(collection, rebuild: bool) -> set[str]:
    """Returns the set of source filenames already present in the collection.
    
    Args:
        collection: ChromaDB collection to inspect.
        rebuild: Whether the collection is being rebuilt from scratch.

    Returns:
        A set of source filenames that are already indexed.
    """
    if rebuild:
        return set()
    existing = collection.get(include=["metadatas"])
    indexed = {m["source"] for m in existing["metadatas"] if m and "source" in m}
    if indexed:
        print(f"[Resume] {len(indexed)} source file(s) already indexed -> skip")
    return indexed


def index_json(
    json_dir: Path,
    db_dir: Path,
    embed_model_name: str = EMBED_MODEL_DEFAULT,
    rebuild: bool = False,
    batch_size: int = 32,
    max_tokens: int | None = None,
) -> None:
    """ Index DoclingDocument JSON checkpoints into ChromaDB.

    Loads DoclingDocument JSON checkpoints produced by ``convert.convert_pdfs``, chunks each document 
    using Docling's ``HybridChunker``, generates embeddings, and writes the chunks and metadata to ChromaDB. 
    Already-indexed sources are skipped unless ``rebuild`` is ``True``.

    Args: 
        json_dir: Directory containing DoclingDocument ``.json`` checkpoints.
        db_dir: Directory containing the persistent ChromaDB database.
        embed_model_name: Name or path of the SentenceTransformer embedding model to use.
        rebuild: Whether to delete the existing ChromaDB collection and rebuild the index from scratch.
        batch_size: Number of chunks to embed in each embedding-model batch.
        max_tokens: hard cap on tokens per chunk. If None (default), it's read from the embedding model's own config 
            (max_seq_length for sentence-transformers models). 
    """
    from docling_core.types.doc.document import DoclingDocument
    from docling.chunking import HybridChunker
    from sentence_transformers import SentenceTransformer
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer 
    from transformers import AutoTokenizer

    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        print(f"No .json files found in {json_dir}")
        return

    print(f"Loading embedding model {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)

    effective_max_tokens = max_tokens or embed_model.get_max_seq_length()
    hf_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embed_model_name),
        max_tokens=effective_max_tokens,
    )
    chunker = HybridChunker(tokenizer=hf_tokenizer)

    collection = _get_or_create_collection(db_dir, rebuild)
    indexed = _already_indexed_sources(collection, rebuild)

    total_chunks = 0
    for json_path in json_paths:
        source_name = f"{json_path.stem}.pdf"
        if source_name in indexed:
            print(f"Skip {json_path.name} (already indexed)")
            continue

        try:
            doc = DoclingDocument.load_from_json(json_path)
        except Exception as e:
            print(f"Load {json_path.name} FAILED ({e})")
            continue

        chunks = list(chunker.chunk(doc))
        if not chunks:
            print(f"Chunk {json_path.name} -> 0 chunks, skipping")
            del doc
            gc.collect()
            continue

        texts = [chunker.contextualize(c) for c in chunks]
        ids = [f"{json_path.stem}::chunk_{i}" for i in range(len(chunks))]
        metadata = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        embeddings = embed_model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadata)

        total_chunks += len(chunks)
        print(f"Index {json_path.name} -> +{len(chunks)} chunks written")

        del doc, chunks, texts, embeddings
        gc.collect()

    print(f"\nDone {total_chunks} new chunks indexed into {db_dir} (collection: {COLLECTION_NAME})")


def index_markdown(
    md_dir: Path,
    db_dir: Path,
    embed_model_name: str = EMBED_MODEL_DEFAULT,
    rebuild: bool = False,
    embed_batch_size: int = 32,
    max_tokens: int = 400,
) -> None:
    """ Index Markdown files into ChromaDB

    Uses an alternative to Docling's ``HybridChunker``. 
    Can be used to process md-Filed obtained from different OCR-model or elsewhere.

    Args:
        md_dir: Directory containing the converted ``.md`` files.
        db_dir: Directory containing the persistent ChromaDB database.
        embed_model_name: Identifier of the SentenceTransformer embedding model to use.
        rebuild: Whether to delete the existing ChromaDB collection and rebuild the index from scratch.
        embed_batch_size: Number of chunks to embed in each embedding-model batch.
        max_tokens: Maximum number of tokenizer tokens per Markdown chunk.
    """
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    md_paths = sorted(md_dir.glob("*.md"))
    if not md_paths:
        print(f"No .md files found in {md_dir}")
        return

    print(f"Loading embedding model {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

    collection = _get_or_create_collection(db_dir, rebuild)
    indexed = _already_indexed_sources(collection, rebuild)

    total_chunks = 0
    for md_path in md_paths:
        source_name = f"{md_path.stem}.pdf" 
        if source_name in indexed:
            print(f"Skip {md_path.name} (already indexed)")
            continue

        text = md_path.read_text(encoding="utf-8")
        chunk_texts = md_chunker(text, tokenizer, max_tokens=max_tokens)
        if not chunk_texts:
            print(f"[chunk] {md_path.name} -> 0 chunks, skipping")
            continue

        ids = [f"{md_path.stem}::chunk_{i}" for i in range(len(chunk_texts))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunk_texts))]

        embeddings = embed_model.encode(
            chunk_texts, batch_size=embed_batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        collection.add(ids=ids, embeddings=embeddings.tolist(), documents=chunk_texts, metadatas=metadatas)

        total_chunks += len(chunk_texts)
        print(f"Index {md_path.name} -> +{len(chunk_texts)} chunks")

    print(f"\nDone - {total_chunks} new chunks indexed into {db_dir} (collection: {COLLECTION_NAME})")

def md_chunker(text: str, tokenizer, max_tokens: int = 400) -> list[str]:
    """ Split Markdown text into token-limited, section-aware chunks.

    The chunker preserves Markdown section context by prefixing each chunk
    with its heading hierarchy. Within each section, paragraphs are greedily
    combined until adding another paragraph would exceed ``max_tokens``. 
    If the document contains no headings, the entire document is treated as a single section.

    Args:
        text: Markdown text to split into chunks.
        tokenizer: Tokenizer used to measure the number of tokens in each paragraph, sentence, and resulting chunk.
        max_tokens: Maximum target number of tokens per chunk.

    Returns:
        A list of text chunks. Each chunk contains its section heading path, when available, 
        followed by the corresponding document content.
    """
    import re 

    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []  # (heading_path, body_lines)
    heading_stack: list[str] = []
    current_body: list[str] = []

    def flush():
        if current_body and "".join(current_body).strip():
            sections.append((" > ".join(heading_stack), current_body.copy()))

    # Split on Markdown headings so each chunk stays within one section
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            current_body.clear()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
        else:
            current_body.append(line)
    flush()

    if not sections:
        sections = [("", lines)]

    def n_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    chunks: list[str] = []
    for heading_path, body_lines in sections:
        prefix = f"{heading_path}\n\n" if heading_path else ""
        paragraphs = [p.strip() for p in "\n".join(body_lines).split("\n\n") if p.strip()]

        buf: list[str] = []
        buf_tokens = 0
        for para in paragraphs:
            p_tokens = n_tokens(para)

            # Paragraph itself too long -> split on sentences.
            if p_tokens > max_tokens:
                if buf:
                    chunks.append(prefix + "\n\n".join(buf))
                    buf, buf_tokens = [], 0
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sbuf, sbuf_tokens = [], 0
                for sent in sentences:
                    st = n_tokens(sent)
                    if sbuf_tokens + st > max_tokens and sbuf:
                        chunks.append(prefix + " ".join(sbuf))
                        sbuf, sbuf_tokens = [], 0
                    sbuf.append(sent)
                    sbuf_tokens += st
                if sbuf:
                    chunks.append(prefix + " ".join(sbuf))
                continue

            if buf_tokens + p_tokens > max_tokens and buf:
                chunks.append(prefix + "\n\n".join(buf))
                buf, buf_tokens = [], 0
            buf.append(para)
            buf_tokens += p_tokens

        if buf:
            chunks.append(prefix + "\n\n".join(buf))

    return chunks

