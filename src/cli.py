"""
Command-line interface. Run with:
    python -m cli <command> ...

USAGE:
  1. Convert PDFs -> DoclingDocument, saved as json-files. Use --device cuda if you have NVIDIA GPU is available:
    python -m cli convert --input ./pdfs --json-out ./docling_json --md-out ./converted_md --device cuda    

  2a. Chunk + embed + index from checkpoints:
    python -m cli index-json --json-dir ./docling_json --db ./chroma_db

  2b. Chunk + embed + index from Markdown file (alternative to Docling's processing pipeline):
    python -m cli index-md --md-dir ./converted_md --db ./chroma_db

  3a. Query the index (retrieval only; --context includes neighboring chunks):
    e.g. python -m cli query --db ./chroma_db --q "How does the paper make use of expected utility?"
    
    Optional: use a reranker. Setup first:
    python -m cli build-bm25 --db chroma_db
    python -m cli query --db ./chroma_db --q "How does the paper make use of utility?" --rerank-model BAAI/bge-reranker-base

  4.  Generate an actual answer from the retrieved chunks (using Ollama):
    e.g. python -m cli answer --db ./chroma_db --q "How does the paper make use of expected utility?"

 The retrieval/generation can be restricted to a single paper:
    python -m cli list-sources --db ./chroma_db
    e.g. python -m cli answer --db ./chroma_db --q "..." --source "ID-123.pdf"
"""

import argparse
from pathlib import Path
from typing import Iterable

from config import EMBED_MODEL_DEFAULT, COLLECTION_NAME
from convert import convert_pdfs
from indexing import index_json, index_markdown
from retrieval import query_index, list_sources
from question_answering import generate_answer

from reranking import build_bm25_index


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Paper RAG search pipeline: convert PDFs, build a vector index, retrieve chunks, and generate LLM answers.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="Convert PDFs -> DoclingDocument, saved as json-files.")
    p_convert.add_argument("--input", type=Path, required=True, help="Directory containing the input PDF files.")
    p_convert.add_argument("--json-out", type=Path, required=True, help="Where to save DoclingDocument .json checkpoints")
    p_convert.add_argument("--md-out", type=Path, default=None, help="Optional: also save readable .md")
    p_convert.add_argument("--rebuild", action="store_true", help="Re-convert even if a checkpoint already exists")
    p_convert.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], 
                           help="Hardware to run Docling's models on. 'cuda' uses NVIDIA GPU.")
    p_convert.add_argument("--num-threads", type=int, default=4, help="CPU threads (used for non-GPU steps)")
    p_convert.add_argument("--ocr-mode", default="auto", choices=["auto", "always", "never"],
                            help="'auto' (default): per-file check, only OCR scanned PDFs. "
                                 "'always': OCR every file. 'never': OCR nothing.")
    p_convert.add_argument("--no-tables", action="store_true", help="Disable table-structure recognition")
    p_convert.add_argument("--image-scale", type=float, default=1.0, help="Page render resolution; lower = faster")

    p_index_json = sub.add_parser("index-json", help="Chunk + embed + index from saved DoclingDocument .json checkpoints")
    p_index_json.add_argument("--json-dir", type=Path, required=True, help="Folder of .json checkpoints from `convert`")
    p_index_json.add_argument("--db", type=Path, required=True, help="Directory for the persistent ChromaDB database.")
    p_index_json.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT, help=(
                            f"SentenceTransformer model used to embed chunks and tokenize them "
                            f"(default: {EMBED_MODEL_DEFAULT})." ),
    )
    p_index_json.add_argument("--rebuild", action="store_true", help="Wipe and re-index everything from scratch")
    p_index_json.add_argument("--embed-batch-size", type=int, default=32, help="Number of chunks embedded in each batch (default: 32).")
    p_index_json.add_argument("--max-tokens", type=int, default=None,           
                           help="Limit on tokens per chunk. Defaults to the embedding model's own max sequence length")

    p_index_md = sub.add_parser("index-md", help="Build the index from already-saved .md files (no PDF/OCR)")
    p_index_md.add_argument("--md-dir", type=Path, required=True, help="Folder of previously converted .md files")
    p_index_md.add_argument("--db", type=Path, required=True, help="Directory for the persistent ChromaDB database.")
    p_index_md.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT, help=(
                            f"SentenceTransformer model used to embed chunks and tokenize them "
                            f"(default: {EMBED_MODEL_DEFAULT})." ),
    )
    p_index_md.add_argument("--rebuild", action="store_true", help="Wipe and re-index everything from scratch")
    p_index_md.add_argument("--embed-batch-size", type=int, default=32, help="Number of chunks embedded in each batch (default: 32).")
    p_index_md.add_argument("--max-tokens", type=int, default=400, help="Max tokens per chunk")

    p_query = sub.add_parser("query", help="Query the vector index (retrieval only, no LLM)")
    p_query.add_argument("--db", type=Path, required=True, help="Directory for the persistent ChromaDB database.")
    p_query.add_argument("--q", required=True, help="Question or search query to use for retrieval.")
    p_query.add_argument("--top-k", type=int, default=5, help="Maximum number of matching chunks to retrieve (default: 5).")
    p_query.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT, help=(
                            f"SentenceTransformer model used to embed chunks and tokenize them "
                            f"(default: {EMBED_MODEL_DEFAULT})."),
    )
    p_query.add_argument("--rerank-model", default=None, help=("Cross-encoder model used to rerank retrieved candidates. "
                                "If omitted, use dense vector retrieval only. Example: BAAI/bge-reranker-base."),
    )
    p_query.add_argument("--candidate-k", type=int, default=30, help=("Number of candidates retrieved from each retrieval method"))
    p_query.add_argument("--context", type=int, default=1, help="Neighboring chunks to include on each side of a hit (0 = off)")
    p_query.add_argument("--max-chars", type=int, default=1200, help="Max characters shown per result")
    p_query.add_argument("--source", default=None, help="Restrict to one source file, e.g. ID-123.pdf")

    p_answer = sub.add_parser("answer", help="Retrieve + ask a LLM to generate an answer")
    p_answer.add_argument("--db", type=Path, required=True, help="Directory for the persistent ChromaDB database.")
    p_answer.add_argument("--q", required=True, help="Question or search query to use for retrieval.")
    p_answer.add_argument("--top-k", type=int, default=5, help="Maximum number of matching chunks to retrieve (default: 5).")
    p_answer.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT, help=(
                            f"SentenceTransformer model used to embed chunks and tokenize them "
                            f"(default: {EMBED_MODEL_DEFAULT})."),
    )
    p_answer.add_argument("--rerank-model", default=None, help=("Cross-encoder model used to rerank retrieved candidates. "
                        "If omitted, use dense vector retrieval only. Example: BAAI/bge-reranker-base."))
    p_answer.add_argument("--candidate-k", type=int, default=30, help=("Number of candidates retrieved from each retrieval method."))
    p_answer.add_argument("--context", type=int, default=1, help="Neighboring chunks to include on each side of a hit (0 = off)")
    p_answer.add_argument("--source", default=None, help="Restrict to one source file, e.g. ID-123.pdf")
    p_answer.add_argument("--llm-model", default="llama3.2",
                           help="Ollama model to generate the answer with (e.g. llama3.2, qwen2.5:3b)")
    p_answer.add_argument("--max-context-chars", type=int, default=3000,
                           help="Max characters of each retrieved chunk sent to the LLM")
    p_answer.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama server address")

    p_bm25 = sub.add_parser("build-bm25", help="Build the BM25 index from an existing ChromaDB index")
    p_bm25.add_argument("--db", type=Path,required=True, help="Directory containing the persistent ChromaDB database")

    p_list_sources = sub.add_parser("list-sources", help="List every source filename currently indexed")
    p_list_sources.add_argument("--db", type=Path, required=True, help="Directory for the persistent ChromaDB database.")

    args = parser.parse_args(argv)

    if args.command == "convert":
        convert_pdfs(
            args.input,
            args.json_out,
            md_out=args.md_out,
            rebuild=args.rebuild,
            device=args.device,
            num_threads=args.num_threads,
            ocr_mode=args.ocr_mode,
            do_table_structure=not args.no_tables,
            image_scale=args.image_scale,
        )
    elif args.command == "index-json":
        index_json(
            args.json_dir,
            args.db,
            embed_model_name=args.embed_model,
            rebuild=args.rebuild,
            batch_size=args.embed_batch_size,
            max_tokens=args.max_tokens
        )
    elif args.command == "index-md":
        index_markdown(
            args.md_dir,
            args.db,
            embed_model_name=args.embed_model,
            rebuild=args.rebuild,
            embed_batch_size=args.embed_batch_size,
            max_tokens=args.max_tokens,
        )
    elif args.command == "build-bm25":
        import chromadb

        client = chromadb.PersistentClient(path=str(args.db))
        collection = client.get_collection(COLLECTION_NAME)

        output_path = args.db / "bm25.pkl"

        build_bm25_index(
            collection=collection,
            output_path=output_path,
        )

        print(f"BM25 index built successfully: {output_path}")
    elif args.command == "query":
        query_index(
            args.q, args.db, embed_model_name=args.embed_model,
            rerank_model_name=args.rerank_model,
            top_k=args.top_k, candidate_k=args.candidate_k, context=args.context, max_chars=args.max_chars, source=args.source, 
        )
    elif args.command == "answer":
        generate_answer(
            query=args.q,
            db_dir=args.db,
            embed_model_name=args.embed_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            context=args.context,
            source=args.source,
            llm_model=args.llm_model,
            max_context_chars=args.max_context_chars,
            ollama_host=args.ollama_host,
            rerank_model_name=args.rerank_model,
        )
    elif args.command == "list-sources":
        list_sources(args.db)


if __name__ == "__main__":
    main()
