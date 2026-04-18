#!/usr/bin/env python3
"""Export static client-side assets for the visual search UI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR = Path("static/assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visual search assets.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compact_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    metadata = paper.get("metadata", {})
    return {
        "id": paper["id"],
        "title": metadata.get("title_en") or paper.get("title"),
        "title_ja": metadata.get("title_ja") or "",
        "abstract": metadata.get("abstract_en") or "",
        "abstract_ja": metadata.get("abstract_ja") or "",
        "url": metadata.get("content_url") or paper.get("url"),
        "authors": metadata.get("authors") or "",
        "content_type": metadata.get("content_type") or "",
        "track_group": metadata.get("track_group") or "",
        "track_name": metadata.get("track_name") or "",
        "session_name": metadata.get("session_name") or "",
        "session_start": metadata.get("session_start") or "",
        "session_room": metadata.get("session_room") or "",
        "doi": metadata.get("doi") or "",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    papers = load_jsonl(args.data_dir / "papers.jsonl")
    embeddings = np.load(args.data_dir / "embeddings.npy").astype("float32")
    projection = np.load(args.data_dir / "projection.npz")
    coords = projection["coords"].astype("float32")
    mean = projection["mean"].astype("float32")
    components = projection["components"].astype("float32")
    meta = json.loads((args.data_dir / "index_meta.json").read_text(encoding="utf-8"))

    if len(papers) != len(embeddings) or len(papers) != len(coords):
        raise ValueError("papers, embeddings, and projection row counts differ")

    papers_path = args.out_dir / "papers.json"
    embeddings_path = args.out_dir / "embeddings.f32"
    coords_path = args.out_dir / "coords.f32"
    projection_path = args.out_dir / "projection.json"
    manifest_path = args.out_dir / "manifest.json"

    papers_path.write_text(
        json.dumps([compact_paper(paper) for paper in papers], ensure_ascii=False),
        encoding="utf-8",
    )
    embeddings.tofile(embeddings_path)
    coords.tofile(coords_path)
    projection_path.write_text(
        json.dumps(
            {
                "mean": mean.tolist(),
                "components": components.tolist(),
                "count": len(papers),
                "dimensions": int(embeddings.shape[1]),
                "model": meta["model"],
                "source_csv": meta.get("source_csv"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": file_sha256(embeddings_path)[:16],
        "files": {
            "papers.json": file_sha256(papers_path),
            "embeddings.f32": file_sha256(embeddings_path),
            "coords.f32": file_sha256(coords_path),
            "projection.json": file_sha256(projection_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {papers_path}")
    print(f"Wrote {embeddings_path}")
    print(f"Wrote {coords_path}")
    print(f"Wrote {projection_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
