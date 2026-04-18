#!/usr/bin/env python3
"""Small web app for visual CHI 2026 embedding search."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from search_index import embed_query, load_index, snippet

load_dotenv()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
WEB_BASE_PATH = os.environ.get("WEB_BASE_PATH", "").strip().rstrip("/")
MAX_TOP_K = int(os.environ.get("VISUAL_SEARCH_MAX_TOP_K", "50"))
STATIC_DIR = Path("static")


@lru_cache(maxsize=1)
def get_state() -> Dict[str, Any]:
    papers, embeddings, meta = load_index(DATA_DIR)
    projection = np.load(DATA_DIR / "projection.npz")
    coords = projection["coords"].astype("float32")
    mean = projection["mean"].astype("float32")
    components = projection["components"].astype("float32")
    if len(coords) != len(papers):
        raise ValueError(f"{len(coords)} projection points but {len(papers)} papers")
    return {
        "papers": papers,
        "embeddings": embeddings,
        "meta": meta,
        "coords": coords,
        "mean": mean,
        "components": components,
        "client": OpenAI(),
    }


def result_payload(paper: Dict[str, Any], score: float, coord: np.ndarray) -> Dict[str, Any]:
    metadata = paper.get("metadata", {})
    title = metadata.get("title_en") or paper.get("title")
    title_ja = metadata.get("title_ja") or ""
    abstract = metadata.get("abstract_en") or ""
    abstract_ja = metadata.get("abstract_ja") or ""
    return {
        "id": paper["id"],
        "score": round(float(score), 6),
        "x": float(coord[0]),
        "y": float(coord[1]),
        "title": title,
        "title_ja": title_ja,
        "url": metadata.get("content_url") or paper.get("url"),
        "authors": metadata.get("authors"),
        "content_type": metadata.get("content_type"),
        "track_group": metadata.get("track_group"),
        "track_name": metadata.get("track_name"),
        "session_name": metadata.get("session_name"),
        "session_start": metadata.get("session_start"),
        "session_room": metadata.get("session_room"),
        "doi": metadata.get("doi"),
        "snippet": snippet(abstract),
        "snippet_ja": snippet(abstract_ja),
    }


async def index(_request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "visual_search.html")


async def health(_request: Request) -> JSONResponse:
    state = get_state()
    return JSONResponse(
        {
            "ok": True,
            "documents": len(state["papers"]),
            "model": state["meta"]["model"],
            "source_csv": state["meta"].get("source_csv"),
        }
    )


async def search(request: Request) -> JSONResponse:
    params = request.query_params
    query = (params.get("q") or "").strip()
    if not query:
        return JSONResponse({"error": "q is required"}, status_code=400)
    top_k = min(max(int(params.get("top_k", "10")), 1), MAX_TOP_K)

    state = get_state()
    query_embedding = embed_query(
        state["client"], query, state["meta"]["model"], state["meta"].get("dimensions")
    )
    scores = state["embeddings"] @ query_embedding
    indexes = np.argpartition(-scores, top_k - 1)[:top_k]
    indexes = indexes[np.argsort(-scores[indexes])]

    query_coord = ((query_embedding - state["mean"]) @ state["components"].T).astype(
        "float32"
    )
    results = [
        result_payload(
            state["papers"][int(index)],
            float(scores[int(index)]),
            state["coords"][int(index)],
        )
        for index in indexes
    ]
    return JSONResponse(
        {
            "query": query,
            "model": state["meta"]["model"],
            "count": len(results),
            "query_point": {"x": float(query_coord[0]), "y": float(query_coord[1])},
            "results": results,
        }
    )


routes = [
    Route(f"{WEB_BASE_PATH}/", index),
    Route(f"{WEB_BASE_PATH}/api/health", health),
    Route(f"{WEB_BASE_PATH}/api/search", search),
]

if WEB_BASE_PATH:
    routes.append(Route(WEB_BASE_PATH, index))

app = Starlette(debug=False, routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
