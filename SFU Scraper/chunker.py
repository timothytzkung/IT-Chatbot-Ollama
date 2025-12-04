#!/usr/bin/env python3
"""
Smarter chunker for TeamDynamix KB articles (for RAG).

Improvements over v1:
- Thin-splits articles by headings (e.g., "OVERVIEW", "PART A:", "PART B:", etc.)
- Removes stray breadcrumb/category lines at the top (e.g., "Students")
- Preserves the closest section heading as metadata AND in the chunk text
- Uses token-aware splitting (~300 tokens) with sliding-window overlap (default 50)
- Applies sliding window *within each section* so steps stay together

Input:  JSON from finalize_kb_fields.py, where each article has:
    - url
    - title
    - tags
    - clean_body_text
    - article_id
    - duplicate_urls

Output: JSONL with one chunk per line:
    {
      "chunk_id": "9838_0",
      "article_id": "9838",
      "url": "...",
      "title": "Re-sharing a Calendar",
      "tags": [...],
      "duplicate_urls": [...],
      "section_heading": "PART A: Receive the Shared Calendar Invitation",
      "text": "chunk content here...",
      "token_count": 295
    }

Usage:
    python chunker.py \
        --input sfu_it_kb_final.json \
        --output sfu_it_kb_chunks.jsonl \
        --max-tokens 300 \
        --overlap 50
"""

import argparse
import json
import re
from typing import List, Dict, Any, Iterable, Tuple


# Breadcrumb-ish category labels that sometimes sneak into the body
BREADCRUMB_LINES = {
    "students",
    "student",
    "faculty",
    "faculty or staff",
    "staff",
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


def normalize_line(line: str) -> str:
    """Normalize a line: strip, collapse spaces, handle nbsp."""
    line = line.replace("\xa0", " ")
    line = line.strip()
    line = re.sub(r"\s{2,}", " ", line)
    return line


def simple_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: splits on whitespace.
    Treats each word-ish unit as a "token".
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return text.split(" ")


def detokenize(tokens: List[str]) -> str:
    """Join tokens back into a string."""
    return " ".join(tokens).strip()


def is_heading(line: str) -> bool:
    """
    Heuristic to detect section headings.

    Examples we want to catch:
    - "OVERVIEW"
    - "DETAILS"
    - "PART A: Receive the Shared Calendar Invitation"
    - "PART B: Re-instate your Shared Calendar"
    - "TROUBLESHOOTING"

    Rules:
    - Short-ish lines
    - Either all caps, or starts with "PART", or ends with ":" (and not too long)
    """
    norm = normalize_line(line)
    if not norm:
        return False

    # Don't treat one-word breadcrumbs as headings
    if norm.lower() in BREADCRUMB_LINES:
        return False

    # Too long? Probably not a heading.
    if len(norm) > 120:
        return False

    # All caps (ignore punctuation)
    letters = re.sub(r"[^A-Za-z]+", "", norm)
    if letters and letters.isupper():
        return True

    # Starts with "Part " or "PART "
    if norm.lower().startswith("part "):
        return True

    # Ends with ":" and not too long
    if norm.endswith(":") and len(norm.split()) <= 10:
        return True

    return False


def split_into_sections(text: str) -> List[Tuple[str, List[str]]]:
    """
    Split an article body into sections based on headings.

    Returns a list of (section_heading, section_lines).

    - Removes breadcrumb-ish lines at the very top.
    - A heading line starts a new section.
    - Lines until the next heading belong to the current section.
    """
    raw_lines = text.splitlines()
    # Normalize but keep original alignment via list indices
    lines = [normalize_line(l) for l in raw_lines]

    # Strip leading breadcrumb lines
    while lines and lines[0].lower() in BREADCRUMB_LINES:
        lines.pop(0)

    sections: List[Tuple[str, List[str]]] = []
    current_heading = ""
    current_lines: List[str] = []

    for line in lines:
        if not line:
            # preserve blank line as separator inside section
            current_lines.append("")
            continue

        if is_heading(line):
            # start new section
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    # Add last section
    if current_lines:
        sections.append((current_heading, current_lines))

    # If no headings at all, treat entire text as one unnamed section
    if not sections:
        sections = [("", [l for l in lines])]

    return sections


def chunk_tokens(
    tokens: List[str],
    max_tokens: int = 300,
    overlap: int = 50,
) -> List[List[str]]:
    """
    Chunk a flat token list into overlapping windows.

    Example:
      max_tokens = 300, overlap = 50

      Chunk 0: tokens[0 : 300]
      Chunk 1: tokens[250 : 550]
      Chunk 2: tokens[500 : 800]
      ...

    Returns a list of token lists.
    """
    if not tokens:
        return []

    if overlap >= max_tokens:
        raise ValueError("overlap must be < max_tokens")

    chunks: List[List[str]] = []
    n = len(tokens)
    start = 0

    while start < n:
        end = min(start + max_tokens, n)
        chunk = tokens[start:end]
        if not chunk:
            break
        chunks.append(chunk)

        if end == n:
            break

        start = end - overlap

    return chunks


def chunk_section(
    article_meta: Dict[str, Any],
    section_heading: str,
    section_lines: List[str],
    max_tokens: int,
    overlap: int,
    chunk_index_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Chunk a single section of an article into one or more chunks.

    Returns:
      - list of chunk dicts
      - next available chunk index (for building chunk_id)
    """
    # Rebuild section text with paragraph breaks
    # (blank lines are already preserved in section_lines)
    section_text = "\n".join(section_lines).strip()
    if not section_text:
        return [], chunk_index_start

    body_tokens = simple_tokenize(section_text)
    token_chunks = chunk_tokens(body_tokens, max_tokens=max_tokens, overlap=overlap)

    chunks: List[Dict[str, Any]] = []
    article_id = str(article_meta.get("article_id") or "")
    url = article_meta.get("url") or ""
    title = article_meta.get("title") or ""
    tags = article_meta.get("tags") or []
    dup_urls = article_meta.get("duplicate_urls") or []

    for token_chunk in token_chunks:
        text = detokenize(token_chunk)
        if not text:
            continue

        # Prepend heading if present and not already in text
        heading = section_heading.strip()
        full_text = text
        if heading:
            # avoid doubling if heading already at start
            if not full_text.lower().startswith(heading.lower()):
                full_text = f"{heading}\n\n{text}"

        chunk_id = f"{article_id}_{chunk_index_start}"
        chunk_index_start += 1

        chunk_obj = {
            "chunk_id": chunk_id,
            "article_id": article_id,
            "url": url,
            "title": title,
            "tags": tags,
            "duplicate_urls": dup_urls,
            "section_heading": heading,
            "text": full_text,
            "token_count": len(token_chunk),
        }
        chunks.append(chunk_obj)

    return chunks, chunk_index_start


def chunk_article(
    article: Dict[str, Any],
    max_tokens: int,
    overlap: int,
) -> Iterable[Dict[str, Any]]:
    """
    Chunk a single article into multiple chunk records.

    Uses section-aware splitting:
    - Split into sections based on headings
    - Chunk each section independently with sliding-window
    """
    body = article.get("clean_body_text") or ""
    if not body.strip():
        return  # nothing to yield

    sections = split_into_sections(body)
    chunk_index = 0
    all_chunks: List[Dict[str, Any]] = []

    for heading, lines in sections:
        section_chunks, chunk_index = chunk_section(
            article_meta=article,
            section_heading=heading,
            section_lines=lines,
            max_tokens=max_tokens,
            overlap=overlap,
            chunk_index_start=chunk_index,
        )
        all_chunks.extend(section_chunks)

    # Yield in order
    for ch in all_chunks:
        yield ch


def main():
    parser = argparse.ArgumentParser(
        description="Chunk KB articles into ~300-token overlapping, heading-aware chunks for RAG.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to finalized KB JSON (e.g. kb_final.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output chunks JSONL (e.g. kb_chunks.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max tokens (words) per chunk (default: 300)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap tokens (words) between consecutive chunks within a section (default: 50)",
    )

    args = parser.parse_args()

    print(f"[main] Loading articles from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        raise ValueError("Input JSON must be a list of article objects.")

    print(f"[main] Loaded {len(articles)} articles.")
    total_chunks = 0

    print(f"[main] Chunking with max_tokens={args.max_tokens}, overlap={args.overlap}...")
    with open(args.output, "w", encoding="utf-8") as out_f:
        for article in articles:
            for chunk in chunk_article(article, args.max_tokens, args.overlap):
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[main] Wrote {total_chunks} chunks to {args.output}.")
    print("[main] Done.")


if __name__ == "__main__":
    main()
