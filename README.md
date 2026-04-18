# CHI 2026 Search

Semantic search, visual exploration, and a remote MCP server for the CHI 2026
program.

This repository builds a lightweight local search index from SIGCHI program
data, serves a browser-based visual explorer, and exposes read-only MCP tools
that ChatGPT can use to search CHI papers conversationally.

## What It Includes

- OpenAI Embeddings based semantic search over CHI 2026 metadata
- Read-only MCP server with `search` and `fetch` tools
- Browser visual search UI with client-side similarity calculation
- Japanese title / abstract / tagline support
- One-process deployment for UI, MCP, static assets, and embedding relay
- Caddy-friendly path deployment, for example `/chi26/`

## Repository Layout

```text
export_program_csv.py          Download SIGCHI program JSON and export CSVs
build_index.py                 Build papers.jsonl and embeddings.npy
search_index.py                CLI semantic search
mcp_server.py                  Unified MCP + visual search server
web_search_app.py              Compatibility entrypoint for the same server
build_visual_assets.py         Export browser-side search assets
build_projection.py            Legacy/projection asset builder
translate_csv.py               Generate Japanese title/abstract with vLLM
generate_taglines.py           Generate Japanese one-line summaries with vLLM
sync_translations_to_papers.py Copy translations into papers.jsonl
sync_taglines_to_papers.py     Copy taglines into papers.jsonl
static/                        Visual search web UI
data/                          Runtime index/cache artifacts
```

## Setup

Install dependencies with `uv`.

```bash
uv sync
```

Create `.env` from `.env.example` and set your OpenAI API key.

```env
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=
DATA_DIR=data

APP_HOST=127.0.0.1
APP_PORT=8000
APP_BASE_PATH=
MCP_PATH=/mcp
APP_ALLOWED_HOSTS=localhost
APP_ALLOWED_ORIGINS=http://localhost:8000
```

For production behind a public domain, set:

```env
APP_ALLOWED_HOSTS=demo.honma.site
APP_ALLOWED_ORIGINS=https://demo.honma.site
```

These host/origin settings are required because FastMCP enables DNS rebinding
protection.

## Build Data

Download the public SIGCHI program data and export CSVs:

```bash
uv run python export_program_csv.py
```

Build the embedding index:

```bash
uv run python build_index.py --input chi2026_program_all_tracks_ja.csv
```

This writes:

```text
data/papers.jsonl
data/embeddings.npy
data/index_meta.json
```

Build browser-side assets:

```bash
uv run python build_visual_assets.py
```

## Run Locally

Start the unified server:

```bash
uv run python mcp_server.py
```

Default endpoints:

```text
http://127.0.0.1:8000/            Visual search UI
http://127.0.0.1:8000/mcp         MCP endpoint
http://127.0.0.1:8000/api/health  Health check
http://127.0.0.1:8000/api/embed   Query embedding relay
```

CLI search:

```bash
uv run python search_index.py "steering model" --top-k 5
```

## MCP Tools

The server exposes two read-only MCP tools.

- `search`: semantic search over CHI 2026 content
- `fetch`: full details for a result by `content_id`

Returned metadata includes titles, abstracts, Japanese fields where available,
authors, DOI, session information, and CHI program URLs.

## Deploy With Caddy

Recommended Caddy configuration for serving under `/chi26/`:

```caddyfile
demo.honma.site {
    redir /chi26 /chi26/

    handle_path /chi26/* {
        reverse_proxy 127.0.0.1:8000
    }
}
```

Use this `.env` shape on the server:

```env
APP_HOST=127.0.0.1
APP_PORT=8000
APP_BASE_PATH=
MCP_PATH=/mcp
APP_ALLOWED_HOSTS=demo.honma.site
APP_ALLOWED_ORIGINS=https://demo.honma.site
```

Public URLs:

```text
https://demo.honma.site/chi26/
https://demo.honma.site/chi26/mcp
```

## Connect From ChatGPT

For an unpublished custom MCP server, enable Developer mode in ChatGPT:

```text
Settings -> Apps & Connectors -> Advanced settings -> Developer mode
```

Then create a connector with:

```text
Connector URL: https://demo.honma.site/chi26/mcp
```

After registration, ChatGPT should show the `search` and `fetch` tools.

## Japanese Translation And Taglines

The translation/tagline scripts target an OpenAI-compatible vLLM endpoint.

```bash
uv run python translate_csv.py --input chi2026_program_all_tracks.csv --in-place
uv run python generate_taglines.py --input chi2026_program_all_tracks_ja.csv --in-place
uv run python sync_translations_to_papers.py
uv run python sync_taglines_to_papers.py
uv run python build_visual_assets.py
```

Relevant environment variables:

```env
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=gemma4-e4b
TRANSLATION_MODEL=gemma4-e4b
TRANSLATION_CONCURRENCY=100
TAGLINE_CONCURRENCY=100
```

Taglines are generated from the original English title and abstract, not from
the Japanese translation.

## Notes

- The MCP server is read-only.
- `OPENAI_API_KEY` stays server-side.
- The visual UI calculates similarity in the browser from static vectors.
- `/api/embed` is public if the app is public; add rate limiting for heavier
  public use.
- Public ChatGPT App submission requires a separate review process. Developer
  mode is enough for private testing.
