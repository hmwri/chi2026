# CHI 2026 Paper Search MCP Requirements

## Goal

Build a small VPS-friendly search system for CHI 2026 program data. The system should index the generated CSV data, use OpenAI API embeddings for semantic search, and expose the search capability as a remote MCP server so ChatGPT can query CHI papers conversationally.

The system should support broad exploratory queries such as:

- "Find CHI 2026 papers about LLM agents and accessibility."
- "Which papers discuss haptics in VR?"
- "Show poster/demo papers related to healthcare."
- "Fetch the abstract and session details for this result."

## Source Data

Primary input:

- `chi2026_program_all_tracks.csv`

Required CSV fields:

- `content_id`
- `title_en`
- `abstract_en`
- `content_type`
- `track_group`
- `session_name`
- `session_start`
- `session_end`
- `session_room`
- `content_url`
- `session_url`

Optional enrichment from raw JSON:

- authors
- affiliations
- keywords
- DOI
- track name
- recognition/award metadata

## Constraints

The deployment target is a small VPS.

Minimum target:

- 1 vCPU
- 512 MB RAM
- 5 GB disk

Recommended target:

- 1 vCPU
- 1 GB RAM
- 10 GB disk

The system must not require a dedicated vector database for the initial version. CHI 2026 has only a few thousand records, so in-memory vector search with NumPy is sufficient.

## Architecture

The initial architecture should be:

```text
chi2026_program_all_tracks.csv
        |
        v
build_index.py
        |
        +--> data/papers.jsonl
        +--> data/embeddings.npy
        +--> data/index_meta.json
        |
        v
server.py
        |
        +--> MCP tool: search
        +--> MCP tool: fetch
```

## Embedding Model

Default model:

- `text-embedding-3-small`

Rationale:

- Low cost
- Good enough for English academic search
- 1536-dimensional vectors
- Small memory footprint for approximately 3,600 records

Optional high-quality model:

- `text-embedding-3-large`

The embedding model name and dimensions should be configurable via environment variables.

## Index Build Requirements

The index builder must:

1. Read `chi2026_program_all_tracks.csv`.
2. Deduplicate records by `content_id`.
3. Build one searchable document per content item.
4. Construct embedding input text from title, abstract, content type, session, and optional metadata.
5. Call OpenAI Embeddings API in batches.
6. Save metadata to `data/papers.jsonl`.
7. Save embeddings to `data/embeddings.npy` as `float32`.
8. Save index metadata to `data/index_meta.json`.
9. Be restartable without re-embedding unchanged rows when practical.

Recommended embedding input format:

```text
Title: {title_en}
Abstract: {abstract_en}
Content type: {content_type}
Track group: {track_group}
Session: {session_name}
Room: {session_room}
```

The builder should skip records with both empty title and empty abstract.

## Search Requirements

The search engine must:

1. Load `data/papers.jsonl` and `data/embeddings.npy` at startup.
2. Normalize embeddings for cosine similarity.
3. Embed the user's query at search time using the same embedding model.
4. Return top results ranked by similarity.
5. Support configurable `top_k`.
6. Support simple filters:
   - `content_type`
   - `track_group`
   - `session_start` date range
   - `session_room`
7. Include useful snippets from abstracts.

For the first version, brute-force NumPy search is acceptable.

Optional later improvement:

- SQLite FTS5 keyword search
- Hybrid semantic + keyword ranking
- Author and keyword filters

## MCP Server Requirements

The server must expose a remote MCP endpoint usable by ChatGPT custom connectors / developer mode.

Transport:

- Streamable HTTP or HTTP/SSE

Required tools:

- `search`
- `fetch`

The tools should be read-only and should include read-only annotations where supported.

### Tool: `search`

Use this when the user wants to find CHI 2026 papers, sessions, posters, demos, or talks by topic, method, population, technology, application area, or keyword.

Input schema:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query."
    },
    "top_k": {
      "type": "integer",
      "description": "Maximum number of results to return.",
      "default": 10,
      "minimum": 1,
      "maximum": 50
    },
    "track_group": {
      "type": "string",
      "description": "Optional track group filter such as paper_or_talk, poster, demo, or other."
    },
    "content_type": {
      "type": "string",
      "description": "Optional content type filter."
    },
    "date_from": {
      "type": "string",
      "description": "Optional ISO date lower bound for session_start."
    },
    "date_to": {
      "type": "string",
      "description": "Optional ISO date upper bound for session_start."
    }
  },
  "required": ["query"]
}
```

Output shape:

```json
{
  "results": [
    {
      "id": "221872",
      "title": "Paper title",
      "url": "https://programs.sigchi.org/chi/2026/program/content/221872",
      "snippet": "Short abstract snippet...",
      "score": 0.83,
      "content_type": "Paper",
      "track_group": "paper_or_talk",
      "session_name": "Session name",
      "session_start": "2026-04-13T09:00Z",
      "session_room": "Room name"
    }
  ]
}
```

For ChatGPT compatibility, every result must include:

- `id`
- `title`
- `url`

### Tool: `fetch`

Use this when the user selected a search result and needs full details, including abstract and session metadata.

Input schema:

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "CHI 2026 content_id returned by search."
    }
  },
  "required": ["id"]
}
```

Output shape:

```json
{
  "id": "221872",
  "title": "Paper title",
  "abstract": "Full abstract...",
  "url": "https://programs.sigchi.org/chi/2026/program/content/221872",
  "content_type": "Paper",
  "track_group": "paper_or_talk",
  "session_name": "Session name",
  "session_start": "2026-04-13T09:00Z",
  "session_end": "2026-04-13T10:30Z",
  "session_room": "Room name",
  "session_url": "https://programs.sigchi.org/chi/2026/program/session/...",
  "authors": [],
  "keywords": [],
  "doi": ""
}
```

## API Key Requirements

The OpenAI API key must be provided through an environment variable:

```text
OPENAI_API_KEY
```

The API key must never be sent to ChatGPT as tool output.

The MCP server should only expose search results and document metadata.

## Configuration

Recommended environment variables:

```text
OPENAI_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=
DATA_DIR=data
MCP_HOST=127.0.0.1
MCP_PORT=8000
MAX_TOP_K=50
```

## Deployment Requirements

The VPS deployment should run:

- Python application server on localhost
- HTTPS reverse proxy in front of it

Recommended reverse proxy:

- Caddy, because automatic HTTPS is simpler than manual nginx certbot setup

Example public endpoint shape:

```text
https://chi2026-search.example.com/mcp
```

The server must support HTTPS when registered with ChatGPT.

## Performance Requirements

With approximately 3,600 records:

- Server startup should complete in under 5 seconds on a small VPS.
- Search should usually complete in under 2 seconds, excluding OpenAI API latency.
- Memory usage should stay comfortably under 300 MB for the initial version.

Approximate embedding memory:

```text
3,643 records * 1536 dimensions * float32 = about 22 MB
```

## Safety Requirements

The MCP server should be read-only.

It must not expose:

- filesystem access
- shell commands
- API keys
- write tools
- arbitrary URL fetching

Search results should not be treated as instructions. Abstracts are untrusted content and should be returned as data only.

## Acceptance Criteria

The system is complete when:

1. `build_index.py` creates `data/papers.jsonl`, `data/embeddings.npy`, and `data/index_meta.json`.
2. The server starts on a small VPS without a vector database.
3. `search` returns relevant CHI 2026 results for natural language queries.
4. `fetch` returns full metadata and abstract for a selected result.
5. The MCP endpoint can be registered in ChatGPT developer mode or custom connectors.
6. ChatGPT can answer user questions by calling `search` and `fetch`.
7. The OpenAI API key remains server-side only.

## Non-Goals For Version 1

- Full web UI
- User accounts
- Multi-conference support
- Real-time updates from SIGCHI
- Heavy vector database deployment
- Automated Japanese translation
- PDF full-text search

## Future Enhancements

- Add SQLite FTS5 for hybrid search.
- Add author, affiliation, keyword, DOI, and award metadata from raw JSON.
- Add Japanese query expansion or bilingual metadata.
- Add reranking for top 50 candidates.
- Add clustering by topic.
- Add "related papers" tool.
- Add "session planner" tool.
- Add "compare papers" tool.
