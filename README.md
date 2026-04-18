# CHI 2026 Search

Local embedding search, visual search UI, and MCP server for CHI 2026 program
data.

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

## Unified Server

Start the unified Streamable HTTP MCP server and visual search app:

```powershell
uv run python mcp_server.py
```

By default it listens on one port:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/mcp
```

Configuration is read from `.env`:

```text
OPENAI_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=
DATA_DIR=data
APP_HOST=127.0.0.1
APP_PORT=8000
APP_BASE_PATH=
MCP_PATH=/mcp
```

Exposed MCP tools:

- `search`: semantic search over CHI 2026 content
- `fetch`: fetch full details by `content_id`

Both tools include Japanese fields where available, including `title_ja`,
`abstract_ja`, and `tagline_ja`.

## Visual Search Web App

Build the 2D projection once after rebuilding embeddings:

```powershell
uv run python build_projection.py
```

Start the same unified app:

```powershell
uv run python mcp_server.py
```

Default local URL:

```text
http://127.0.0.1:8000/
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
        reverse_proxy 127.0.0.1:8000
    }
}
```

With this `handle_path` setup, keep:

```text
APP_BASE_PATH=
MCP_PATH=/mcp
```

The public MCP URL is:

```text
https://example.com/chi2026/mcp
```

If you do not strip the prefix, set `APP_BASE_PATH` and `MCP_PATH`, then proxy
the path as-is:

```text
APP_BASE_PATH=/chi2026
MCP_PATH=/chi2026/mcp
```

```caddyfile
example.com {
    redir /chi2026 /chi2026/

    handle /chi2026/* {
        reverse_proxy 127.0.0.1:8000
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

## Japanese Tagline Generation With vLLM

Generate short Japanese one-line summaries separately from translation:

```powershell
uv run python generate_taglines.py --input chi2026_program_all_tracks_ja.csv
```

This writes `tagline_ja` to a CSV and caches results in:

```text
data/taglines_ja.json
```

The tagline prompt is written in Japanese. It always uses `title_en` and
`abstract_en` as input so the tagline is generated from the original English
paper text rather than from the Japanese translation.

Useful options:

```powershell
uv run python generate_taglines.py --dry-run --limit 3
uv run python generate_taglines.py --limit 10 --output chi2026_program_all_tracks_ja_tagline_test.csv
uv run python generate_taglines.py --in-place
uv run python generate_taglines.py --concurrency 100
uv run python generate_taglines.py --strict
```

To copy cached taglines into `data/papers.jsonl`:

```powershell
uv run python sync_taglines_to_papers.py
```
