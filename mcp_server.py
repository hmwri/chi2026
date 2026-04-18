#!/usr/bin/env python3
"""MCP server for the local CHI 2026 embedding search index."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from search_index import embed_query, load_index, snippet

load_dotenv()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
MAX_TOP_K = int(os.environ.get("MAX_TOP_K", "50"))

mcp = FastMCP(
    "CHI 2026 Paper Search",
    instructions=(
        "Search and fetch CHI 2026 program papers, posters, demos, sessions, "
        "authors, abstracts, DOI links, and presentation metadata. All tools "
        "are read-only."
    ),
    stateless_http=True,
    json_response=True,
    host=MCP_HOST,
    port=MCP_PORT,
)


@lru_cache(maxsize=1)
def get_search_state() -> Dict[str, Any]:
    papers, embeddings, meta = load_index(DATA_DIR)
    papers_by_id = {paper["id"]: paper for paper in papers}
    return {
        "papers": papers,
        "papers_by_id": papers_by_id,
        "embeddings": embeddings,
        "meta": meta,
        "client": OpenAI(),
    }


def parse_date(value: str) -> str:
    return (value or "").strip()[:10]


def paper_matches_filters(
    paper: Dict[str, Any],
    track_group: Optional[str],
    content_type: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> bool:
    metadata = paper.get("metadata", {})
    if track_group and metadata.get("track_group") != track_group:
        return False
    if content_type:
        current = (metadata.get("content_type") or "").lower()
        display = (metadata.get("content_type_display_name") or "").lower()
        wanted = content_type.lower()
        if wanted not in current and wanted not in display:
            return False
    session_date = parse_date(metadata.get("session_start", ""))
    if date_from and session_date and session_date < date_from:
        return False
    if date_to and session_date and session_date > date_to:
        return False
    return True


def result_from_paper(paper: Dict[str, Any], score: float) -> Dict[str, Any]:
    metadata = paper.get("metadata", {})
    return {
        "id": paper["id"],
        "title": metadata.get("title_en") or paper.get("title"),
        "title_ja": metadata.get("title_ja"),
        "tagline_ja": metadata.get("tagline_ja"),
        "url": metadata.get("content_url") or paper.get("url"),
        "snippet": snippet(metadata.get("abstract_en", "")),
        "snippet_ja": snippet(metadata.get("abstract_ja", "")),
        "score": round(float(score), 6),
        "content_type": metadata.get("content_type"),
        "track_group": metadata.get("track_group"),
        "track_name": metadata.get("track_name"),
        "authors": metadata.get("authors"),
        "doi": metadata.get("doi"),
        "session_name": metadata.get("session_name"),
        "session_start": metadata.get("session_start"),
        "session_end": metadata.get("session_end"),
        "session_room": metadata.get("session_room"),
    }


@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    track_group: Optional[str] = None,
    content_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Search CHI 2026 content by natural language query.

    Use this for broad exploratory searches over titles, abstracts, authors,
    affiliations, sessions, events, and metadata. Results are ranked by cosine
    similarity against the local OpenAI embedding index.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    state = get_search_state()
    papers: List[Dict[str, Any]] = state["papers"]
    embeddings: np.ndarray = state["embeddings"]
    meta: Dict[str, Any] = state["meta"]

    requested_top_k = max(1, min(int(top_k), MAX_TOP_K))
    query_embedding = embed_query(
        state["client"], query, meta["model"], meta.get("dimensions")
    )
    scores = embeddings @ query_embedding

    candidate_indexes = [
        index
        for index, paper in enumerate(papers)
        if paper_matches_filters(paper, track_group, content_type, date_from, date_to)
    ]
    if not candidate_indexes:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "model": meta["model"],
            "message": "No documents matched the provided filters.",
        }

    candidate_scores = scores[candidate_indexes]
    top_k = min(requested_top_k, len(candidate_indexes))
    top_positions = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
    top_positions = top_positions[np.argsort(-candidate_scores[top_positions])]

    results = []
    for position in top_positions:
        paper_index = candidate_indexes[int(position)]
        results.append(result_from_paper(papers[paper_index], float(scores[paper_index])))

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "total_documents": len(papers),
        "model": meta["model"],
    }


@mcp.tool()
def fetch(id: str) -> Dict[str, Any]:
    """Fetch full details for a CHI 2026 content item by content_id."""
    content_id = str(id).strip()
    if not content_id:
        raise ValueError("id must not be empty")

    state = get_search_state()
    paper = state["papers_by_id"].get(content_id)
    if paper is None:
        return {"id": content_id, "found": False}

    metadata = paper.get("metadata", {})
    return {
        "id": paper["id"],
        "found": True,
        "title": metadata.get("title_en") or paper.get("title"),
        "title_ja": metadata.get("title_ja"),
        "tagline_ja": metadata.get("tagline_ja"),
        "abstract": metadata.get("abstract_en"),
        "abstract_ja": metadata.get("abstract_ja"),
        "url": metadata.get("content_url") or paper.get("url"),
        "authors": metadata.get("authors"),
        "affiliations": metadata.get("affiliations"),
        "author_countries": metadata.get("author_countries"),
        "keywords": metadata.get("keywords"),
        "doi": metadata.get("doi"),
        "doi_url": metadata.get("doi_url"),
        "presentation_video_url": metadata.get("presentation_video_url"),
        "content_type": metadata.get("content_type"),
        "track_group": metadata.get("track_group"),
        "track_name": metadata.get("track_name"),
        "award": metadata.get("award"),
        "session_name": metadata.get("session_name"),
        "session_start": metadata.get("session_start"),
        "session_end": metadata.get("session_end"),
        "session_room": metadata.get("session_room"),
        "session_floor": metadata.get("session_floor"),
        "session_chairs": metadata.get("session_chairs"),
        "session_url": metadata.get("session_url"),
        "content_events": metadata.get("content_events"),
        "event_start": metadata.get("event_start"),
        "event_end": metadata.get("event_end"),
        "event_rooms": metadata.get("event_rooms"),
        "event_floors": metadata.get("event_floors"),
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
