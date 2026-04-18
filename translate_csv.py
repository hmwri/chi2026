#!/usr/bin/env python3
"""Translate CHI 2026 CSV title/abstract columns into Japanese via vLLM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_INPUT = Path("chi2026_program_all_tracks.csv")
DEFAULT_CACHE = Path("data/translations_ja.json")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gemma4-e4b"
DEFAULT_CONCURRENCY = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate title_ja and abstract_ja columns in a CHI CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input stem>_ja.csv unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV after writing a .bak backup.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TRANSLATION_MODEL", os.environ.get("VLLM_MODEL", DEFAULT_MODEL)
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="API key for OpenAI-compatible server. vLLM often accepts EMPTY.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("TRANSLATION_CONCURRENCY", DEFAULT_CONCURRENCY)),
        help="Number of concurrent translation requests.",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", "").split())


def content_hash(title: str, abstract: str) -> str:
    payload = json.dumps(
        {"title_en": title, "abstract_en": abstract},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def ensure_fields(fieldnames: List[str]) -> List[str]:
    out = list(fieldnames)
    if "title_ja" not in out:
        insert_at = out.index("title_en") + 1 if "title_en" in out else len(out)
        out.insert(insert_at, "title_ja")
    if "abstract_ja" not in out:
        insert_at = out.index("abstract_en") + 1 if "abstract_en" in out else len(out)
        out.insert(insert_at, "abstract_ja")
    return out


def read_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_path(args: argparse.Namespace) -> Path:
    if args.in_place:
        return args.input
    if args.output is not None:
        return args.output
    return args.input.with_name(f"{args.input.stem}_ja{args.input.suffix}")


def cache_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    content_id = clean_text(row.get("content_id"))
    title = clean_text(row.get("title_en"))
    abstract = clean_text(row.get("abstract_en"))
    return content_id, title, abstract, content_hash(title, abstract)


def needs_translation(row: Dict[str, str], overwrite: bool) -> bool:
    title = clean_text(row.get("title_en"))
    abstract = clean_text(row.get("abstract_en"))
    if not title and not abstract:
        return False
    if overwrite:
        return True
    return not (clean_text(row.get("title_ja")) and clean_text(row.get("abstract_ja")))


def extract_json(text: str) -> Dict[str, str]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    return {
        "title_ja": clean_text(data.get("title_ja")),
        "abstract_ja": clean_text(data.get("abstract_ja")),
    }


def translate_one(
    base_url: str,
    api_key: str,
    model: str,
    title: str,
    abstract: str,
    max_retries: int = 3,
) -> Dict[str, str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    system = (
        "You are a professional academic translator. Translate the provided "
        "CHI paper title and abstract into natural Japanese. Preserve technical "
        "terms, acronyms, product names, citations, and numbers. Return only "
        "valid JSON with keys title_ja and abstract_ja."
    )
    user = json.dumps(
        {"title_en": title, "abstract_en": abstract},
        ensure_ascii=False,
        indent=2,
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content or ""
            return extract_json(content)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(2**attempt, 10)
            print(f"Translation failed ({exc}); retrying in {delay}s...", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def translate_work_item(
    item: Tuple[str, str, str, str],
    base_url: str,
    api_key: str,
    model: str,
    sleep_seconds: float,
) -> Tuple[str, Dict[str, str]]:
    content_id, title, abstract, hash_value = item
    translated = translate_one(base_url, api_key, model, title, abstract)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return content_id, {
        "hash": hash_value,
        "title_ja": translated["title_ja"],
        "abstract_ja": translated["abstract_ja"],
    }


def build_unique_work(
    rows: List[Dict[str, str]],
    cache: Dict[str, Dict[str, str]],
    overwrite: bool,
) -> List[Tuple[str, str, str, str]]:
    work = []
    seen = set()
    for row in rows:
        if not needs_translation(row, overwrite):
            continue
        content_id, title, abstract, hash_value = cache_key(row)
        if not content_id or hash_value in seen:
            continue
        if not overwrite and cache.get(content_id, {}).get("hash") == hash_value:
            continue
        seen.add(hash_value)
        work.append((content_id, title, abstract, hash_value))
    return work


def apply_cache_to_rows(
    rows: List[Dict[str, str]], cache: Dict[str, Dict[str, str]], overwrite: bool
) -> None:
    for row in rows:
        content_id, _title, _abstract, hash_value = cache_key(row)
        cached = cache.get(content_id)
        if not cached or cached.get("hash") != hash_value:
            continue
        if overwrite or not clean_text(row.get("title_ja")):
            row["title_ja"] = cached.get("title_ja", "")
        if overwrite or not clean_text(row.get("abstract_ja")):
            row["abstract_ja"] = cached.get("abstract_ja", "")


def backup_input(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_bytes(path.read_bytes())
    return backup


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        return 1

    rows, fieldnames = read_csv(args.input)
    fieldnames = ensure_fields(fieldnames)
    cache = read_cache(args.cache)
    work = build_unique_work(rows, cache, args.overwrite)
    if args.limit is not None:
        work = work[: args.limit]

    print(f"Rows: {len(rows)}")
    print(f"Unique translations to request: {len(work)}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Concurrency: {args.concurrency}")

    if args.dry_run:
        if work:
            content_id, title, abstract, _hash_value = work[0]
            print(
                json.dumps(
                    {
                        "content_id": content_id,
                        "title_en": title,
                        "abstract_en_preview": abstract[:500],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0

    if args.concurrency < 1:
        print("--concurrency must be at least 1.", file=sys.stderr)
        return 1

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                translate_work_item,
                item,
                args.base_url,
                args.api_key,
                args.model,
                args.sleep,
            )
            for item in work
        ]
        for future in as_completed(futures):
            content_id, cached = future.result()
            cache[content_id] = cached
            completed += 1
            if completed % 10 == 0:
                write_cache(args.cache, cache)
            print(f"Translated {completed}/{len(work)}: {content_id}")

    write_cache(args.cache, cache)
    apply_cache_to_rows(rows, cache, args.overwrite)

    out_path = output_path(args)
    if args.in_place:
        backup = backup_input(args.input)
        print(f"Backed up input to {backup}")
    write_csv(out_path, fieldnames, rows)
    print(f"Wrote {out_path}")
    print(f"Wrote {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
