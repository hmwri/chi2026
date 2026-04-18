#!/usr/bin/env python3
"""Static visual search app plus a tiny OpenAI embedding relay."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

load_dotenv()

WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
WEB_BASE_PATH = os.environ.get("WEB_BASE_PATH", "").strip().rstrip("/")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = (
    int(os.environ["EMBEDDING_DIMENSIONS"])
    if os.environ.get("EMBEDDING_DIMENSIONS")
    else None
)
STATIC_DIR = Path("static")

client = OpenAI()


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


async def index(_request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "visual_search.html")


async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "model": EMBEDDING_MODEL})


async def embed(request: Request) -> JSONResponse:
    query = (request.query_params.get("q") or "").strip()
    if not query:
        return JSONResponse({"error": "q is required"}, status_code=400)

    kwargs: Dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "input": query,
        "encoding_format": "float",
    }
    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS

    response = client.embeddings.create(**kwargs)
    embedding = normalize_l2(
        np.array(response.data[0].embedding, dtype="float32")
    ).astype("float32")
    return JSONResponse(
        {
            "query": query,
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "embedding": embedding.tolist(),
        }
    )


routes = [
    Route(f"{WEB_BASE_PATH}/", index),
    Route(f"{WEB_BASE_PATH}/api/health", health),
    Route(f"{WEB_BASE_PATH}/api/embed", embed),
    Mount(
        f"{WEB_BASE_PATH}/assets",
        app=StaticFiles(directory=STATIC_DIR / "assets"),
        name="assets",
    ),
]

if WEB_BASE_PATH:
    routes.append(Route(WEB_BASE_PATH, index))

app = Starlette(debug=False, routes=routes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
