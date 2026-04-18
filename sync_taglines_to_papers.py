#!/usr/bin/env python3
"""Copy cached Japanese taglines into data/papers.jsonl metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PAPERS = Path("data/papers.jsonl")
DEFAULT_CACHE = Path("data/taglines_ja.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync tagline_ja from tagline cache to papers.jsonl."
    )
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak backup before updating papers.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if not args.papers.exists():
        raise FileNotFoundError(args.papers)
    if not args.cache.exists():
        raise FileNotFoundError(args.cache)

    papers = load_jsonl(args.papers)
    taglines = json.loads(args.cache.read_text(encoding="utf-8"))

    updated = 0
    missing = 0
    empty = 0
    for paper in papers:
        content_id = str(paper.get("id", ""))
        generated = taglines.get(content_id)
        if not generated:
            missing += 1
            continue
        tagline_ja = clean_text(generated.get("tagline_ja"))
        if not tagline_ja:
            empty += 1
            continue
        paper.setdefault("metadata", {})["tagline_ja"] = tagline_ja
        updated += 1

    print(f"papers: {len(papers)}")
    print(f"taglines: {len(taglines)}")
    print(f"updated: {updated}")
    print(f"missing: {missing}")
    print(f"empty: {empty}")

    if args.dry_run:
        return 0

    if not args.no_backup:
        backup = args.papers.with_suffix(args.papers.suffix + ".bak")
        backup.write_bytes(args.papers.read_bytes())
        print(f"Backed up {args.papers} to {backup}")
    write_jsonl(args.papers, papers)
    print(f"Wrote {args.papers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
