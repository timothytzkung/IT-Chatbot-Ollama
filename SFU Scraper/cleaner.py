#!/usr/bin/env python3
"""
Clean scraped TeamDynamix KB data for RAG.

- Removes portal UI chrome, navigation, feedback widgets, etc.
- Strips duplicate titles and 'Article - ...' heading lines.
- Extracts tags from "Tags" section inside body_text (bounded, safe).
- Produces `clean_body_text` suitable for chunking/embedding, with a
  fallback if cleaning is too aggressive.

Usage:
    python cleaner.py \
        --input kb_ragset_out.json \
        --output kb_ragset_out_clean.json
"""

import argparse
import json
import re
from typing import List, Dict, Any, Set


# Lines that should be dropped if they match exactly (case-insensitive)
EXACT_TRASH_LINES = {
    "sfu.ca",
    "skip to main content",
    "(opens in a new tab)",
    "filter your search by category. current category:",
    "all",
    "knowledge base",
    "service catalog",
    "search the client portal",
    "search",
    "sign in",
    "show applications menu",
    "its client portal",
    "home",
    "services",
    "more applications",
    "skip to knowledge base content",
    "articles",
    "blank",
    "sign in to leave feedback",
    "0 reviews",
    "print article",
    "deleting...",
    "article",

    # SFU-specific category/breadcrumb-ish labels we don't want in text
    "updating...",
    "faculty or staff",
    "desktop and mobile computing",
    "printing and related services",
    "sfu print",
    "communication and collaboration",
    "email and collaboration services",
    "sfu mail",
    "migration information",
    "conferencing and telephones",
    "ms teams phone",
}

# Lines that should be dropped if they contain these substrings (case-insensitive)
CONTAINS_TRASH_SUBSTRINGS = [
    "show applications menu",
    "search the client portal",
    "knowledge base content",
    "client portal",
    "was this article helpful",
]


def normalize_line(line: str) -> str:
    """Normalize a single line: strip, collapse spaces, handle nbsp."""
    line = line.replace("\xa0", " ").strip()
    line = re.sub(r"\s{2,}", " ", line)
    return line


def should_drop_line(line: str, title: str, tag_lines_norm: Set[str]) -> bool:
    """
    Determine if a line is boilerplate/trash that should be removed.

    `line` is expected to already be normalized (normalize_line).
    """
    if not line:
        return False

    norm = line.lower()
    title_norm = title.strip().lower()

    # Drop lines that are just the title (we already know the title)
    if title_norm and norm == title_norm:
        return True

    # Drop "Article - <something>" header lines
    if norm.startswith("article - "):
        return True

    # Drop tag section label & tag lines (they live in metadata instead)
    if norm == "tags":
        return True
    if norm in tag_lines_norm:
        return True

    # Exact junk lines
    if norm in EXACT_TRASH_LINES:
        return True

    # Lines that contain junk substrings
    for sub in CONTAINS_TRASH_SUBSTRINGS:
        if sub in norm:
            return True

    return False


def extract_tag_lines(lines: List[str]) -> List[str]:
    """
    Given all body_text lines, extract lines that represent tags
    after a 'Tags' label.

    Pattern we expect:

        Tags
        tag1
        tag2
        ...
        Overview / Details / etc.

    We:
    - look for a line 'Tags' (case-insensitive)
    - collect following non-empty lines
    - stop at the first likely heading ('Overview', 'Details', etc.)
      or when we hit obvious UI/boilerplate
    - also cap number of tag lines and max words per tag to avoid
      runaway extraction where body text becomes tags.
    """
    tags: List[str] = []
    tag_start_idx = None

    # Find the "Tags" line
    for idx, raw in enumerate(lines):
        line = normalize_line(raw)
        if line.lower() == "tags":
            tag_start_idx = idx + 1
            break

    if tag_start_idx is None:
        return tags  # none found

    MAX_TAG_LINES = 20       # safety cap
    MAX_TAG_WORDS = 5        # tag lines shouldn't be full sentences
    HEADING_STOP_WORDS = {"overview", "details", "introduction", "summary"}

    for i in range(tag_start_idx, len(lines)):
        raw = lines[i]
        line = normalize_line(raw)
        lower = line.lower()

        # stop on blank line
        if not line:
            break

        # stop when we hit a typical section heading
        if lower in HEADING_STOP_WORDS:
            break

        # stop if it looks like obvious UI junk
        if lower in EXACT_TRASH_LINES:
            break

        # safety caps
        if len(tags) >= MAX_TAG_LINES:
            break
        if len(line.split()) > MAX_TAG_WORDS:
            # very long lines are likely real content, not tags
            break

        tags.append(line)

    return tags


def extract_and_merge_tags(article: Dict[str, Any]) -> List[str]:
    """
    Return an updated tags list, merging existing tags with any discovered
    tags in the body_text 'Tags' section.
    """
    existing_tags = article.get("tags") or []
    body_text = article.get("body_text") or ""

    lines = [l for l in body_text.splitlines()]
    discovered_tags = extract_tag_lines(lines)

    # Normalize / dedupe
    tags_set = set(t.strip() for t in existing_tags if t)
    merged: List[str] = list(existing_tags)

    for t in discovered_tags:
        if t and t not in tags_set:
            merged.append(t)
            tags_set.add(t)

    return merged


def _clean_lines_core(
    raw_lines: List[str],
    title: str,
    drop_tags: bool = True,
) -> str:
    """
    Core cleaner over a list of raw lines.
    If drop_tags=False, we ignore the 'Tags' handling (used in fallback).
    """
    # Determine tag lines if enabled
    tag_lines_norm: Set[str] = set()
    if drop_tags:
        tag_lines_raw = extract_tag_lines(raw_lines)
        tag_lines_norm = {normalize_line(t).lower() for t in tag_lines_raw}

    cleaned_lines: List[str] = []

    for raw_line in raw_lines:
        line = normalize_line(raw_line)

        # Skip lines that are pure junk or tag lines
        if drop_tags and should_drop_line(line, title, tag_lines_norm):
            continue
        if not drop_tags:
            # Even in fallback, we still drop heavy portal chrome & duplicate titles
            norm = line.lower()
            title_norm = title.strip().lower()
            if title_norm and norm == title_norm:
                continue
            if norm.startswith("article - "):
                continue
            if norm in EXACT_TRASH_LINES:
                continue
            for sub in CONTAINS_TRASH_SUBSTRINGS:
                if sub in norm:
                    line = ""  # mark as droppable
                    break
            if not line:
                continue

        cleaned_lines.append(line)

    # Collapse multiple consecutive blank lines into a single blank line
    final_lines: List[str] = []
    last_was_blank = False

    for line in cleaned_lines:
        if not line:
            if not last_was_blank:
                final_lines.append("")
                last_was_blank = True
        else:
            final_lines.append(line)
            last_was_blank = False

    # Strip leading/trailing blank lines
    while final_lines and final_lines[0] == "":
        final_lines.pop(0)
    while final_lines and final_lines[-1] == "":
        final_lines.pop()

    return "\n".join(final_lines).strip()


def clean_body_text(article: Dict[str, Any]) -> str:
    """
    Clean the body_text of a single article with a robust strategy:

    1. Run full cleaning (drop tags from body, drop portal chrome, etc.)
    2. If that seems to have over-cleaned (tiny output but large original),
       fall back to a lighter clean that only removes obvious chrome.
    """
    title = article.get("title", "") or ""
    body_text = article.get("body_text", "") or ""
    raw_lines = body_text.splitlines()

    # Full clean (with tag removal)
    clean_text = _clean_lines_core(raw_lines, title=title, drop_tags=True)

    # Fallback heuristic:
    # If the cleaned text is very short but the original is long,
    # we likely over-cleaned (e.g., nuked the entire article as "tags").
    if len(clean_text) < 120 and len(body_text) > 500:
        print(
            "[warn] Very small clean_body_text vs large body_text; "
            "using fallback clean for article:",
            title,
        )
        clean_text = _clean_lines_core(raw_lines, title=title, drop_tags=False)

    return clean_text


def process_articles(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process all articles, adding `clean_body_text` and improved `tags`."""
    processed: List[Dict[str, Any]] = []
    empty_count = 0

    for article in data:
        # Merge tags from body_text if needed (bounded/safe)
        merged_tags = extract_and_merge_tags(article)
        article["tags"] = merged_tags

        clean_text = clean_body_text(article)
        article["clean_body_text"] = clean_text

        if not clean_text:
            empty_count += 1

        processed.append(article)

    print(f"[clean] Processed {len(processed)} articles.")
    print(f"[clean] {empty_count} articles ended up with empty clean_body_text.")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Clean scraped TeamDynamix KB JSON for RAG."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to scraped JSON file (e.g. teamdynamix_kb.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output cleaned JSON file (e.g. teamdynamix_kb_clean.json)",
    )

    args = parser.parse_args()

    print(f"[main] Loading scraped data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of article objects.")

    print(f"[main] Loaded {len(data)} articles.")
    cleaned = process_articles(data)

    print(f"[main] Writing cleaned data to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("[main] Done.")


if __name__ == "__main__":
    main()
