#!/usr/bin/env python3
"""Generate short Japanese paper taglines for a CHI CSV via vLLM."""
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

DEFAULT_INPUT = Path("chi2026_program_all_tracks_ja.csv")
DEFAULT_CACHE = Path("data/taglines_ja.json")
DEFAULT_FAILURE_LOG = Path("data/tagline_failures.jsonl")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gemma4-e4b"
DEFAULT_CONCURRENCY = 100


class TaglineParseError(ValueError):
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tagline_ja in a CHI CSV using title and abstract."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input stem>_tagline.csv unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV after writing a .bak backup.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_FAILURE_LOG)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TAGLINE_MODEL",
            os.environ.get("TRANSLATION_MODEL", os.environ.get("VLLM_MODEL", DEFAULT_MODEL)),
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
        default=int(os.environ.get("TAGLINE_CONCURRENCY", DEFAULT_CONCURRENCY)),
        help="Number of concurrent tagline requests.",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first failed generation instead of continuing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", "").split())


def content_hash(title: str, abstract: str) -> str:
    payload = json.dumps(
        {"title": title, "abstract": abstract},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_fields(fieldnames: List[str]) -> List[str]:
    out = list(fieldnames)
    if "tagline_ja" not in out:
        insert_at = out.index("abstract_ja") + 1 if "abstract_ja" in out else len(out)
        out.insert(insert_at, "tagline_ja")
    return out


def output_path(args: argparse.Namespace) -> Path:
    if args.in_place:
        return args.input
    if args.output is not None:
        return args.output
    return args.input.with_name(f"{args.input.stem}_tagline{args.input.suffix}")


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


def append_failure(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def cache_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    content_id = clean_text(row.get("content_id"))
    title = clean_text(row.get("title_en"))
    abstract = clean_text(row.get("abstract_en"))
    return content_id, title, abstract, content_hash(title, abstract)


def needs_tagline(row: Dict[str, str], overwrite: bool) -> bool:
    _content_id, title, abstract, _hash_value = cache_key(row)
    if not title and not abstract:
        return False
    if overwrite:
        return True
    return not clean_text(row.get("tagline_ja"))


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def load_json_lenient(text: str) -> Dict[str, Any]:
    candidates = [strip_code_fence(text)]
    match = re.search(r"\{.*\}", candidates[0], flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise TaglineParseError(f"Could not parse model output as JSON: {last_error}", text)


def extract_json(text: str) -> Dict[str, str]:
    data = load_json_lenient(text.strip())
    return {"tagline_ja": clean_text(data.get("tagline_ja"))}


def generate_one(
    base_url: str,
    api_key: str,
    model: str,
    title: str,
    abstract: str,
    max_retries: int = 3,
) -> Dict[str, str]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    system = (
        "あなたはHCI論文を素早く理解するための日本語キャッチコピーを作る編集者です。"
        "入力された論文タイトルとアブストラクトを読み、何をした研究なのかが一目でわかる"
        "短い日本語のひとことまとめを作ってください。"
        "文字数はおおむね20文字程度にしてください。厳密な文字数よりも、具体性と自然さを優先します。"
        "単なる分野名や題目ではなく、研究の行為・貢献・検証内容が伝わる表現にしてください。"
        "「新しい研究」「提案手法」「HCI研究」「ユーザー調査」のような抽象的すぎる表現は避けてください。"
        "句点は不要です。出力は必ずJSONのみで、キーはtagline_jaだけにしてください。"
    )
    user = json.dumps(
        {"title": title, "abstract": abstract},
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
                temperature=0.2,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            return extract_json(content)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(2**attempt, 10)
            print(f"Tagline generation failed ({exc}); retrying in {delay}s...", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def generate_work_item(
    item: Tuple[str, str, str, str],
    base_url: str,
    api_key: str,
    model: str,
    sleep_seconds: float,
) -> Tuple[str, Dict[str, str]]:
    content_id, title, abstract, hash_value = item
    generated = generate_one(base_url, api_key, model, title, abstract)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return content_id, {
        "hash": hash_value,
        "tagline_ja": generated["tagline_ja"],
    }


def build_unique_work(
    rows: List[Dict[str, str]],
    cache: Dict[str, Dict[str, str]],
    overwrite: bool,
) -> List[Tuple[str, str, str, str]]:
    work = []
    seen = set()
    for row in rows:
        if not needs_tagline(row, overwrite):
            continue
        content_id, title, abstract, hash_value = cache_key(row)
        if not content_id or hash_value in seen:
            continue
        cached = cache.get(content_id, {})
        if (
            not overwrite
            and cached.get("hash") == hash_value
            and clean_text(cached.get("tagline_ja"))
        ):
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
        if overwrite or not clean_text(row.get("tagline_ja")):
            row["tagline_ja"] = cached.get("tagline_ja", "")


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
    print(f"Unique taglines to request: {len(work)}")
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
                        "title": title,
                        "abstract_preview": abstract[:500],
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
    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for item in work:
            content_id, title, abstract, hash_value = item
            future = executor.submit(
                generate_work_item,
                item,
                args.base_url,
                args.api_key,
                args.model,
                args.sleep,
            )
            futures[future] = {
                "content_id": content_id,
                "title": title,
                "abstract_hash": hash_value,
                "abstract_preview": abstract[:500],
            }

        for future in as_completed(futures):
            context = futures[future]
            content_id = context["content_id"]
            try:
                _content_id, cached = future.result()
            except Exception as exc:
                failed += 1
                failure = {
                    **context,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_output": getattr(exc, "raw_output", "")[:4000],
                    "model": args.model,
                    "base_url": args.base_url,
                    "created_at_unix": int(time.time()),
                }
                append_failure(args.failure_log, failure)
                print(
                    f"Failed {failed} / pending item {content_id}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if args.strict:
                    raise
            else:
                cache[_content_id] = cached
                completed += 1
                if completed % 10 == 0:
                    write_cache(args.cache, cache)
                print(f"Generated {completed}/{len(work)}: {_content_id}")

    write_cache(args.cache, cache)
    apply_cache_to_rows(rows, cache, args.overwrite)

    out_path = output_path(args)
    if args.in_place:
        backup = backup_input(args.input)
        print(f"Backed up input to {backup}")
    write_csv(out_path, fieldnames, rows)
    print(f"Wrote {out_path}")
    print(f"Wrote {args.cache}")
    if failed:
        print(f"Failed taglines: {failed}")
        print(f"Wrote {args.failure_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
