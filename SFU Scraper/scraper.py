#!/usr/bin/env python3
"""
Scrape a TeamDynamix Knowledge Base into JSON for use in a RAG pipeline.

Usage:
    python scraper.py \
        --kb-root "https://sfu.teamdynamix.com/TDClient/255/ITServices/KB/" \
        --domain "sfu.teamdynamix.com" \
        --output "kb_ragset_out.json"
"""

import argparse
import json
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def is_same_domain(url: str, domain: str) -> bool:
    """Check if URL is on the same domain."""
    try:
        return urlparse(url).netloc == domain
    except Exception:
        return False


def crawl_kb(
    start_url: str,
    domain: str,
    delay: float = 0.5,
    session: requests.Session | None = None,
) -> set[str]:
    """
    Breadth-first crawl of the TeamDynamix KB area, collecting all ArticleDet URLs.

    - Only follows links under /TDClient/.../KB/
    - Collects any URL containing 'ArticleDet?ID='
    """
    if session is None:
        session = requests.Session()

    visited: set[str] = set()
    article_urls: set[str] = set()
    queue: list[str] = [start_url]

    print(f"[crawl] Starting crawl at: {start_url}")

    while queue:
        url = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"[crawl] Fetching: {url}")
            resp = session.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"[crawl] Error fetching {url}: {e}")
            continue

        if resp.status_code != 200:
            print(f"[crawl] Non-200 status {resp.status_code} for {url}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all links on the page
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])

            # Same domain only
            if not is_same_domain(href, domain):
                continue

            # Only stay under KB path
            # (You can tighten this if you know the exact path)
            if "/TDClient/" in href and "/KB/" in href:
                # Article detail pages look like .../ArticleDet?ID=1234
                if "ArticleDet?ID=" in href:
                    article_urls.add(href)
                else:
                    # Likely category / search / navigation page
                    if href not in visited:
                        queue.append(href)

        time.sleep(delay)  # be polite

    print(f"[crawl] Done. Found {len(article_urls)} article URLs.")
    return article_urls


def scrape_article(
    url: str,
    session: requests.Session | None = None,
    delay: float = 0.2,
) -> dict:
    """
    Scrape a single TeamDynamix KB article page.

    Tries to extract:
    - title (h1)
    - tags (links near a 'Tags' label)
    - body_html (HTML between the title and 'Details' header, if present)
    - body_text (plain text version)
    - (you can add created/modified if you want to parse them later)
    """
    if session is None:
        session = requests.Session()

    print(f"[article] Scraping: {url}")
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[article] Error fetching {url}: {e}")
        raise

    soup = BeautifulSoup(resp.text, "html.parser")

    # ---------- TITLE ----------
    # Often the article title is the main H1 element.
    # You can tighten this selector if your portal uses a specific ID/class
    # e.g. soup.select_one("h1#tdxArticleSubject")
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # ---------- TAGS ----------
    # TeamDynamix often shows tags near a "Tags" label.
    # This is heuristic – inspect your portal and adjust as needed.
    tags: list[str] = []
    tag_container = None

    # Look for a string containing "Tag" or "Tags" in strong/h2/h3
    for el in soup.find_all(["strong", "b", "h2", "h3"]):
        text = el.get_text(strip=True)
        if "Tag" in text:  # catches "Tag(s)", "Tags:", etc.
            tag_container = el.parent
            break

    if tag_container:
        for a in tag_container.find_all("a", href=True):
            tag_text = a.get_text(strip=True)
            if tag_text:
                tags.append(tag_text)

    # ---------- BODY ----------
    # Strategy:
    # - Find the "Details" heading, if it exists
    # - Take everything after the title up to (but not including) the Details header
    # This avoids pulling in the metadata box at the bottom.
    body_html = ""
    body_text = ""

    details_header = None
    for el in soup.find_all(["h2", "h3"]):
        text = el.get_text(strip=True)
        if "Detail" in text:  # "Details", "Article Details", etc.
            details_header = el
            break

    body_container = []

    if title_el and details_header:
        for sibling in title_el.find_all_next():
            if sibling == details_header:
                break
            body_container.append(sibling)

    # Fallback if we couldn't find that structure:
    if not body_container:
        # Try common article container IDs/classes used by many TDX templates.
        # You will likely need to inspect your HTML and adjust this.
        possible_ids = ["tdxArticle", "tdxContent", "contentDiv"]
        main = None
        for pid in possible_ids:
            main = soup.find(id=pid)
            if main:
                break
        if not main:
            # As a last resort, just use the whole <article> tag if exists
            main = soup.find("article")

        if main:
            body_html = str(main)
            body_text = main.get_text("\n", strip=True)
        else:
            # Very last resort: use the entire page body text.
            body_text = soup.get_text("\n", strip=True)
            body_html = ""
    else:
        body_html = "".join(str(s) for s in body_container)
        tmp = BeautifulSoup(body_html, "html.parser")
        body_text = tmp.get_text("\n", strip=True)

    # ---------- (OPTIONAL) DETAILS ----------
    # If you want created/modified dates or article ID from the details table,
    # inspect your portal and add specific parsing here.
    created = None
    modified = None
    article_id = None

    # Example: look for lines that start with "Created", "Modified", "Article ID"
    # (This is very rough; customize as needed.)
    page_text_lines = soup.get_text("\n").splitlines()
    for line in map(str.strip, page_text_lines):
        lower = line.lower()
        if lower.startswith("created") and created is None:
            # e.g. "Created 5/9/2018"
            created = line
        elif lower.startswith("modified") and modified is None:
            modified = line
        elif "article id" in lower and article_id is None:
            article_id = line

    time.sleep(delay)

    return {
        "url": url,
        "title": title,
        "tags": tags,
        "body_html": body_html,
        "body_text": body_text,
        "created_raw": created,
        "modified_raw": modified,
        "article_id_raw": article_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scrape a TeamDynamix Knowledge Base into JSON."
    )
    parser.add_argument(
        "--kb-root",
        required=True,
        help="Root URL of the KB, e.g. https://youruni.teamdynamix.com/TDClient/123/Portal/KB/",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name, e.g. youruni.teamdynamix.com",
    )
    parser.add_argument(
        "--output",
        default="teamdynamix_kb.json",
        help="Output JSON file path (default: teamdynamix_kb.json)",
    )
    parser.add_argument(
        "--crawl-delay",
        type=float,
        default=0.5,
        help="Delay between crawl requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--article-delay",
        type=float,
        default=0.2,
        help="Delay between article requests in seconds (default: 0.2)",
    )

    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "tdx-kb-scraper/1.0 (+for-university-rag-bot)",
        }
    )

    # 1) Crawl to get all article URLs
    article_urls = crawl_kb(
        start_url=args.kb_root,
        domain=args.domain,
        delay=args.crawl_delay,
        session=session,
    )

    # 2) Scrape each article
    articles_data: list[dict] = []
    for idx, url in enumerate(sorted(article_urls), start=1):
        try:
            article = scrape_article(
                url=url,
                session=session,
                delay=args.article_delay,
            )
            articles_data.append(article)
        except Exception as e:
            print(f"[main] Skipping {url} due to error: {e}")

        print(f"[main] Progress: {idx}/{len(article_urls)}")

    # 3) Save to JSON
    print(f"[main] Saving {len(articles_data)} articles to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, ensure_ascii=False, indent=2)

    print("[main] Done.")


if __name__ == "__main__":
    main()
