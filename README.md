# CHI 2026 Search

Local embedding search and MCP server for CHI 2026 program data.

## Setup

Create `.env` from `.env.example` and set `OPENAI_API_KEY`.

```powershell
uv sync
```

## Build The Index

```powershell
uv run python build_index.py
```

This writes:

- `data/papers.jsonl`
- `data/embeddings.npy`
- `data/index_meta.json`

## CLI Search

```powershell
uv run python search_index.py "LLM agents accessibility" --top-k 5
```

## MCP Server

Start the Streamable HTTP MCP server:

```powershell
uv run python mcp_server.py
```

By default it listens on:

```text
http://127.0.0.1:8000/mcp
```

Configuration is read from `.env`:

```text
OPENAI_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=
DATA_DIR=data
MCP_HOST=127.0.0.1
MCP_PORT=8000
```

Exposed MCP tools:

- `search`: semantic search over CHI 2026 content
- `fetch`: fetch full details by `content_id`

## Japanese Translation With vLLM

Start a vLLM OpenAI-compatible server separately, then run:

```powershell
uv run python translate_csv.py --input chi2026_program_all_tracks.csv
```

By default this writes:

```text
chi2026_program_all_tracks_ja.csv
```

The script fills `title_ja` and `abstract_ja`, caches translations in
`data/translations_ja.json`, and avoids retranslating duplicate `content_id`
entries.

Useful options:

```powershell
uv run python translate_csv.py --dry-run --limit 3
uv run python translate_csv.py --limit 10 --output chi2026_program_all_tracks_ja_test.csv
uv run python translate_csv.py --in-place
uv run python translate_csv.py --concurrency 100
```
