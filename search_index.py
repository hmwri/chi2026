#!/usr/bin/env python3
"""Search the local CHI 2026 embedding index from the command line."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DEFAULT_TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search CHI 2026 papers.")
    parser.add_argument("query", help="Search query, such as 'LLM agents accessibility'.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_index(data_dir: Path) -> tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    papers_path = data_dir / "papers.jsonl"
    embeddings_path = data_dir / "embeddings.npy"
    meta_path = data_dir / "index_meta.json"

    missing = [path for path in (papers_path, embeddings_path, meta_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing index files: {names}. Run build_index.py first.")

    papers = load_jsonl(papers_path)
    embeddings = np.load(embeddings_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if len(papers) != len(embeddings):
        raise ValueError(
            f"Index mismatch: {len(papers)} papers but {len(embeddings)} embeddings."
        )
    return papers, embeddings, meta


def embed_query(
    client: OpenAI, query: str, model: str, dimensions: Optional[int]
) -> np.ndarray:
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": query,
        "encoding_format": "float",
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    response = client.embeddings.create(**kwargs)
    return normalize_l2(np.array(response.data[0].embedding, dtype="float32"))


def snippet(text: str, max_len: int = 280) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def print_result(rank: int, score: float, paper: Dict[str, Any]) -> None:
    metadata = paper.get("metadata", {})
    title = metadata.get("title_en") or paper.get("title") or "(untitled)"
    authors = metadata.get("authors") or ""
    content_type = metadata.get("content_type") or ""
    session = metadata.get("session_name") or ""
    url = metadata.get("content_url") or paper.get("url") or ""
    abstract = metadata.get("abstract_en") or ""

    print(f"{rank}. {title}")
    print(f"   score: {score:.4f}")
    if authors:
        print(f"   authors: {authors}")
    if content_type or session:
        print(f"   type/session: {content_type} / {session}")
    if url:
        print(f"   url: {url}")
    if abstract:
        print(f"   abstract: {snippet(abstract)}")
    print()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        print("--top-k must be at least 1.", file=sys.stderr)
        return 1

    papers, embeddings, meta = load_index(args.data_dir)
    model = meta["model"]
    dimensions = meta.get("dimensions")

    client = OpenAI()
    query_embedding = embed_query(client, args.query, model, dimensions)
    scores = embeddings @ query_embedding

    top_k = min(args.top_k, len(papers))
    indexes = np.argpartition(-scores, top_k - 1)[:top_k]
    indexes = indexes[np.argsort(-scores[indexes])]

    print(f"Query: {args.query}")
    print(f"Model: {model}")
    print(f"Results: {top_k}/{len(papers)}")
    print()

    for rank, index in enumerate(indexes, start=1):
        print_result(rank, float(scores[index]), papers[int(index)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
