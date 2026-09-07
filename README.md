# paper-rag-search
RAG pipeline for searching and querying a research paper collection (PDF → text extraction → vector search →  LLM answers)

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

For answer generation, install [Ollama](https://ollama.com/download) and pull
a small local model:

```bash
ollama pull llama3.2
```

Ollama runs a local server automatically (e.g. http://localhost:11434).

## Usage

All commands are run as `python -m cli <command> ...` from
the repo root.

### Convert PDFs to DoclingDocument

```bash
python -m cli convert \
  --input ./pdfs --json-out ./docling_json --md-out ./converted_md --device cuda
```

- `--device cuda` uses NVIDIA GPU (also accepts `cpu` or `auto`).
- OCR is applied automatically only to PDFs that need it (`--ocr-mode auto`). Use `--ocr-mode always` or `--ocr-mode never` to override.
- `--md-out` is optional to also save a readable `.md` copy of each paper.
- Each PDF's parsed result is saved to disk immediately after conversion; Re-running skips anything already converted.
  Use `--rebuild` to force re-conversion.

### Build the vector index

From the DoclingDocument checkpoints:
```bash
python -m cli index-json --json-dir ./docling_json --db ./chroma_db
```

Or alternatively from plain Markdown files: 
```bash
python -m cli index-md --md-dir ./converted_md --db ./chroma_db
```
Both building processes are resumable. Re-running skips files that where already indexed. Use `--rebuild` to discard and rebuild the index from scratch.
 
 ### Query the index 
 Retrieval of relevant chunks only, no answer is generated. 
 - `--context` (default 1) includes neighboring chunks on each side of a hit to broaden context
 - `--top-k` sets how many results are shown.

```bash
python -m cli query --db ./chroma_db --q "How does the paper make use of expected utility?"
```

### Generate answers
Retrieves relevant chunks and asks a Ollama model to write an answer. 
The result includes the source file and chunk range that where consulted.

```bash
python -m cli answer --db ./chroma_db --q "How does the paper make use of expected utility?"
```

### Restrict sources
List all sources available in the collection
```bash
python -m cli list-sources --db ./chroma_db
```

`--source` works on both `query` and `answer`, restricting retrieval to one specific PDF.
```bash
python -m cli answer --db ./chroma_db --q "..." --source "ID-123.pdf"
python -m cli answer --db ./chroma_db --q "..." --source "ID-123.pdf"
```

## File structure 
```
src/
├── config.py               # shared constants (embedding model, collection name, system prompt)
├── convert.py               # PDF -> DoclingDocument JSON-files
├── indexing.py               # chunk + embed + write to ChromaDB
├── retrieval.py               # query + list-sources
├── question_answering.py      # LLM answer generation (Ollama)
└── cli.py                     # argparse wiring
```

