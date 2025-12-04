#!/usr/bin/env python3
"""
Deduplicate TeamDynamix KB articles by canonical article ID (ID=...).

- Groups entries that share the same ID param in the URL
  (e.g. ?ID=10056 and ?ID=10056&SIDs=1162).
- Keeps one canonical article object per ID.
- Merges tags across duplicates.
- Keeps the URL from the first article as `url`.
- Stores all other URLs for the same ID in `duplicate_urls`.
- Prefers the longest `clean_body_text` among duplicates.

Usage:
    python nodupe.py \
        --input kb_ragset_out_clean.json \
        --output kb_ragset_out_clean_nodupe.json
"""

import argparse
import json
import re
from typing import List, Dict, Any


def canonical_article_id(url: str) -> str:
    """
    Extract the TeamDynamix article ID from the URL.

    Example:
        https://.../ArticleDet?ID=10056&SIDs=1470 -> "10056"
    """
    m = re.search(r"[?&]ID=(\d+)", url)
    return m.group(1) if m else url  # fallback: use full URL if no ID param


def dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group articles by canonical article ID and merge duplicates.

    Rules:
    - First article seen for an ID becomes the base.
    - `url` is taken from the first article.
    - `duplicate_urls` contains all other URLs for the same ID.
    - `tags` is the union of all tags across duplicates.
    - `clean_body_text` is taken from the article with the longest body.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for art in articles:
        url = art.get("url", "")
        aid = canonical_article_id(url)

        if aid not in groups:
            # Start a new group with a shallow copy so we don't mutate the input
            base = dict(art)
            base["article_id"] = aid
            base["duplicate_urls"] = []
            base["tags"] = list(base.get("tags") or [])
            groups[aid] = base
        else:
            base = groups[aid]

            # ---- URLs ----
            if url and url != base.get("url") and url not in base["duplicate_urls"]:
                base["duplicate_urls"].append(url)

            # ---- Tags ----
            existing_tags = base.get("tags") or []
            tags_set = set(t for t in existing_tags if t)
            for t in art.get("tags") or []:
                if t and t not in tags_set:
                    existing_tags.append(t)
                    tags_set.add(t)
            base["tags"] = existing_tags

            # ---- clean_body_text: prefer the longest version ----
            cb_old = (base.get("clean_body_text") or "").strip()
            cb_new = (art.get("clean_body_text") or "").strip()
            if len(cb_new) > len(cb_old):
                base["clean_body_text"] = cb_new

    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate TeamDynamix KB JSON by article ID."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned JSON file (e.g. teamdynamix_kb_clean.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output deduped JSON file (e.g. teamdynamix_kb_clean_deduped.json)",
    )

    args = parser.parse_args()

    print(f"[main] Loading cleaned data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of article objects.")

    print(f"[main] Loaded {len(data)} articles.")
    deduped = dedupe_articles(data)
    print(f"[main] Deduped {len(data)} -> {len(deduped)} articles.")

    print(f"[main] Writing deduped data to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print("[main] Done.")


if __name__ == "__main__":
    main()
