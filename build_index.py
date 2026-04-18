#!/usr/bin/env python3
"""Build a local embedding index from the CHI 2026 CSV.

Outputs:
  data/papers.jsonl
  data/embeddings.npy
  data/index_meta.json

The script stores L2-normalized float32 embeddings so search can use a simple
dot product for cosine similarity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CSV = Path("chi2026_program_all_tracks.csv")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CHARS = 24_000

TEXT_FIELDS = [
    ("Title", "title_en"),
    ("Japanese title", "title_ja"),
    ("Abstract", "abstract_en"),
    ("Japanese abstract", "abstract_ja"),
    ("Authors", "authors"),
    ("Affiliations", "affiliations"),
    ("Author departments", "author_departments"),
    ("Author cities", "author_cities"),
    ("Author countries", "author_countries"),
    ("Keywords", "keywords"),
    ("Content type", "content_type"),
    ("Track group", "track_group"),
    ("Track", "track_name"),
    ("Award", "award"),
    ("Tags", "tags"),
    ("Session", "session_name"),
    ("Session description", "session_description"),
    ("Session chairs", "session_chairs"),
    ("Session room", "session_room"),
    ("Session floor", "session_floor"),
    ("Events", "content_events"),
    ("Event descriptions", "event_descriptions"),
    ("Event presenters", "event_presenters"),
    ("Event chairs", "event_chairs"),
]

METADATA_FIELDS = [
    "content_id",
    "title_en",
    "title_ja",
    "abstract_en",
    "abstract_ja",
    "authors",
    "affiliations",
    "author_departments",
    "author_cities",
    "author_states",
    "author_countries",
    "keywords",
    "doi",
    "doi_url",
    "presentation_video_url",
    "content_type",
    "content_type_display_name",
    "track_group",
    "track_name",
    "award",
    "tags",
    "session_name",
    "session_start",
    "session_end",
    "session_room",
    "session_floor",
    "session_chairs",
    "content_events",
    "event_start",
    "event_end",
    "event_rooms",
    "event_floors",
    "content_url",
    "session_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenAI embedding index files from CHI 2026 CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--model", default=os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=int(os.environ["EMBEDDING_DIMENSIONS"])
        if os.environ.get("EMBEDDING_DIMENSIONS")
        else None,
        help="Optional embeddings dimensions parameter.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache and re-embed every document.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and prepare documents without calling the OpenAI API.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", "").split())


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_content_id(row: Dict[str, str]) -> str:
    return clean_text(row.get("content_id"))


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    by_id: Dict[str, Dict[str, str]] = {}
    for row in rows:
        content_id = row_content_id(row)
        title = clean_text(row.get("title_en"))
        abstract = clean_text(row.get("abstract_en"))
        if not content_id or not (title or abstract):
            continue
        current = by_id.get(content_id)
        if current is None:
            by_id[content_id] = row
            continue
        current_score = len(clean_text(current.get("abstract_en"))) + len(
            clean_text(current.get("authors"))
        )
        row_score = len(abstract) + len(clean_text(row.get("authors")))
        if row_score > current_score:
            by_id[content_id] = row
    return sorted(by_id.values(), key=lambda row: row_content_id(row))


def build_search_text(row: Dict[str, str], max_chars: int) -> str:
    parts = []
    for label, field in TEXT_FIELDS:
        value = clean_text(row.get(field))
        if value:
            parts.append(f"{label}: {value}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def build_paper(row: Dict[str, str], max_chars: int) -> Dict[str, Any]:
    metadata = {field: clean_text(row.get(field)) for field in METADATA_FIELDS}
    content_id = metadata["content_id"]
    search_text = build_search_text(row, max_chars)
    document_hash = sha256_text(stable_json({"id": content_id, "text": search_text}))
    return {
        "id": content_id,
        "title": metadata["title_en"],
        "url": metadata["content_url"],
        "metadata": metadata,
        "search_text": search_text,
        "document_hash": document_hash,
    }


def build_papers(csv_path: Path, max_chars: int) -> List[Dict[str, Any]]:
    rows = dedupe_rows(read_rows(csv_path))
    return [build_paper(row, max_chars) for row in rows]


def normalize_l2(matrix: "Any") -> "Any":
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def papers_path(data_dir: Path) -> Path:
    return data_dir / "papers.jsonl"


def embeddings_path(data_dir: Path) -> Path:
    return data_dir / "embeddings.npy"


def meta_path(data_dir: Path) -> Path:
    return data_dir / "index_meta.json"


def load_existing_cache(
    data_dir: Path, model: str, dimensions: Optional[int]
) -> Dict[Tuple[str, str], List[float]]:
    import numpy as np

    paper_file = papers_path(data_dir)
    embedding_file = embeddings_path(data_dir)
    index_meta_file = meta_path(data_dir)
    if not (paper_file.exists() and embedding_file.exists() and index_meta_file.exists()):
        return {}

    meta = json.loads(index_meta_file.read_text(encoding="utf-8"))
    if meta.get("model") != model or meta.get("dimensions") != dimensions:
        return {}

    old_papers = []
    with paper_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                old_papers.append(json.loads(line))

    old_embeddings = np.load(embedding_file)
    if len(old_papers) != len(old_embeddings):
        return {}

    cache = {}
    for paper, embedding in zip(old_papers, old_embeddings):
        cache[(paper["id"], paper["document_hash"])] = embedding.astype("float32")
    return cache


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def embed_batch(
    client: Any,
    texts: Sequence[str],
    model: str,
    dimensions: Optional[int],
    max_retries: int = 5,
) -> List[List[float]]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": list(texts),
        "encoding_format": "float",
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(**kwargs)
            return [item.embedding for item in response.data]
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(2**attempt, 30)
            print(
                f"Embedding batch failed ({exc}); retrying in {delay}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def create_embeddings(
    papers: List[Dict[str, Any]],
    data_dir: Path,
    model: str,
    dimensions: Optional[int],
    batch_size: int,
    force: bool,
) -> "Any":
    import numpy as np
    from openai import OpenAI

    client = OpenAI()
    cache = {} if force else load_existing_cache(data_dir, model, dimensions)
    embeddings: List[Optional[Any]] = [None] * len(papers)
    missing_indexes = []

    for index, paper in enumerate(papers):
        cached = cache.get((paper["id"], paper["document_hash"]))
        if cached is not None:
            embeddings[index] = cached
        else:
            missing_indexes.append(index)

    if missing_indexes:
        print(f"Embedding {len(missing_indexes)} documents with {model}...")
    else:
        print("Reusing cached embeddings for all documents.")

    completed = len(papers) - len(missing_indexes)
    for batch_indexes in batched(missing_indexes, batch_size):
        texts = [papers[index]["search_text"] for index in batch_indexes]
        batch_embeddings = embed_batch(client, texts, model, dimensions)
        for index, embedding in zip(batch_indexes, batch_embeddings):
            embeddings[index] = np.array(embedding, dtype="float32")
        completed += len(batch_indexes)
        print(f"Embedded {completed}/{len(papers)}")

    matrix = np.vstack(embeddings).astype("float32")
    return normalize_l2(matrix).astype("float32")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_outputs(
    data_dir: Path,
    papers: List[Dict[str, Any]],
    embeddings: "Any",
    model: str,
    dimensions: Optional[int],
    source_csv: Path,
) -> None:
    import numpy as np

    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(papers_path(data_dir), papers)
    np.save(embeddings_path(data_dir), embeddings)

    meta = {
        "source_csv": str(source_csv),
        "model": model,
        "dimensions": dimensions,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_normalized": True,
        "document_count": len(papers),
        "created_at_unix": int(time.time()),
    }
    meta_path(data_dir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    papers = build_papers(args.csv, args.max_chars)
    if not papers:
        print("No indexable papers found.", file=sys.stderr)
        return 1

    print(f"Prepared {len(papers)} unique content records from {args.csv}")
    print(f"Model: {args.model}")
    if args.dimensions is not None:
        print(f"Dimensions: {args.dimensions}")

    if args.dry_run:
        sample = {
            "id": papers[0]["id"],
            "title": papers[0]["title"],
            "search_text": papers[0]["search_text"][:1000],
        }
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return 0

    embeddings = create_embeddings(
        papers=papers,
        data_dir=args.data_dir,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        force=args.force,
    )
    write_outputs(args.data_dir, papers, embeddings, args.model, args.dimensions, args.csv)
    print(f"Wrote {papers_path(args.data_dir)}")
    print(f"Wrote {embeddings_path(args.data_dir)}")
    print(f"Wrote {meta_path(args.data_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
