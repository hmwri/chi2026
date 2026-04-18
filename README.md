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

## Visual Search Web App

Build the 2D projection once after rebuilding embeddings:

```powershell
uv run python build_projection.py
```

Start the visual search app:

```powershell
uv run python web_search_app.py
```

Default local URL:

```text
http://127.0.0.1:8080/
```

The app keeps `OPENAI_API_KEY` on the server. The server only relays query
embedding requests to OpenAI via `/api/embed`. The browser loads static
metadata/vectors and performs similarity search and click-to-nearest-paper
exploration locally using `Float32Array` in a Web Worker. The map uses Apache
ECharts with the Canvas renderer.

Build browser-side assets after rebuilding embeddings:

```powershell
uv run python build_visual_assets.py
```

### Caddy Under A Path

Recommended Caddy setup strips the public path prefix before proxying:

```caddyfile
example.com {
    redir /chi2026 /chi2026/

    handle_path /chi2026/* {
        reverse_proxy 127.0.0.1:8080
    }
}
```

With this `handle_path` setup, keep:

```text
WEB_BASE_PATH=
```

If you do not strip the prefix, set `WEB_BASE_PATH` and proxy the path as-is:

```text
WEB_BASE_PATH=/chi2026
```

```caddyfile
example.com {
    redir /chi2026 /chi2026/

    handle /chi2026/* {
        reverse_proxy 127.0.0.1:8080
    }
}
```

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

If one translation returns malformed JSON or otherwise fails, the script keeps
going and records the failed item in:

```text
data/translation_failures.jsonl
```

Use `--strict` if you want the script to stop on the first failure.

Useful options:

```powershell
uv run python translate_csv.py --dry-run --limit 3
uv run python translate_csv.py --limit 10 --output chi2026_program_all_tracks_ja_test.csv
uv run python translate_csv.py --in-place
uv run python translate_csv.py --concurrency 100
uv run python translate_csv.py --strict
```
