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

    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].astype("float32")
    coords = (centered @ components.T).astype("float32")

    np.savez_compressed(
        output_path,
        coords=coords,
        mean=mean.reshape(-1).astype("float32"),
        components=components,
    )
    print(f"Wrote {output_path}")
    print(f"coords shape: {coords.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
