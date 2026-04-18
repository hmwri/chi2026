#!/usr/bin/env python3
"""Build a lightweight 2D PCA projection for the embedding index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DEFAULT_DATA_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D projection data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--method",
        choices=["pca", "umap"],
        default="umap",
        help="2D projection method.",
    )
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    args = parse_args()
    embeddings_path = args.data_dir / "embeddings.npy"
    papers_path = args.data_dir / "papers.jsonl"
    output_path = args.data_dir / "projection.npz"

    embeddings = np.load(embeddings_path).astype("float32")
    papers = load_jsonl(papers_path)
    if len(papers) != len(embeddings):
        raise ValueError(f"{len(papers)} papers but {len(embeddings)} embeddings")

    mean = None
    components = None
    if args.method == "pca":
        mean = embeddings.mean(axis=0, keepdims=True).astype("float32")
        centered = embeddings - mean
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].astype("float32")
        coords = (centered @ components.T).astype("float32")
    else:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.random_state,
            low_memory=True,
        )
        coords = reducer.fit_transform(embeddings).astype("float32")

    payload = {"coords": coords, "method": np.array(args.method)}
    if mean is not None and components is not None:
        payload["mean"] = mean.reshape(-1).astype("float32")
        payload["components"] = components
    np.savez_compressed(output_path, **payload)
    print(f"Wrote {output_path}")
    print(f"method: {args.method}")
    print(f"coords shape: {coords.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
