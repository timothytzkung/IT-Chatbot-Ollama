#!/usr/bin/env python3
"""
Finalize TeamDynamix KB JSON for RAG.

Keeps only:
- url
- title
- tags
- clean_body_text
- article_id
- duplicate_urls

Removes all other fields.

Usage:
    python finalizer.py \
        --input kb_ragset_out_clean_nodupe.json \
        --output sfu_it_kb_final.json
"""

import argparse
import json
from typing import List, Dict, Any


KEEP_FIELDS = {
    "url",
    "title",
    "tags",
    "clean_body_text",
    "article_id",
    "duplicate_urls",
}


def finalize_articles(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a minimal version of each article with only KEEP_FIELDS."""
    finalized = []

    for article in data:
        new_obj = {}

        for key in KEEP_FIELDS:
            # graceful handling: default duplicates array if missing
            if key == "duplicate_urls":
                new_obj[key] = article.get(key, [])
            else:
                new_obj[key] = article.get(key)

        finalized.append(new_obj)

    return finalized


def main():
    parser = argparse.ArgumentParser(
        description="Reduce KB JSON down to only essential RAG fields."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned + deduped JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output final JSON",
    )

    args = parser.parse_args()

    print(f"[main] Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of article objects.")

    print(f"[main] Loaded {len(data)} articles. Finalizing...")
    finalized = finalize_articles(data)

    print(f"[main] Writing: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(finalized, f, ensure_ascii=False, indent=2)

    print("[main] Done.")


if __name__ == "__main__":
    main()
