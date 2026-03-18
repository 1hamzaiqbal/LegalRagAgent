"""Lightweight web scraper for enriching web search results.

Uses trafilatura for fetch + text extraction (highest accuracy among
open-source extractors, F1=0.958). Strips boilerplate/nav/ads automatically.

Used by the pipeline's web_search action type to go beyond DuckDuckGo snippets.

Usage (standalone testing):
  uv run python web_scraper.py "https://www.law.cornell.edu/wex/negligence"
  uv run python web_scraper.py "https://example.com" --max-chars 5000
"""

import sys
import trafilatura


def scrape_url(url: str, max_chars: int = 8000) -> dict:
    """Fetch a URL and extract main text content.

    Args:
        url: The URL to scrape.
        max_chars: Max characters of text to return (truncates from end).

    Returns:
        dict with keys: url, title, text, error (None on success).
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return {"url": url, "title": "", "text": "", "error": "fetch failed"}

        text = trafilatura.extract(
            downloaded,
            include_links=False,
            include_images=False,
            include_tables=True,
            favor_recall=True,  # extract more content vs precision
        )

        if not text:
            return {"url": url, "title": "", "text": "", "error": "no content extracted"}

        # Extract title from metadata
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else ""

        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated]"

        return {"url": url, "title": title, "text": text, "error": None}

    except Exception as e:
        return {"url": url, "title": "", "text": "", "error": str(e)}


def scrape_urls(urls: list[str], max_results: int = 3, **kwargs) -> list[dict]:
    """Scrape multiple URLs, returning results for those with content.

    Stops after max_results successful scrapes. Skips URLs that error or
    return too-short text.
    """
    results = []
    for url in urls:
        if len(results) >= max_results:
            break
        result = scrape_url(url, **kwargs)
        if result["text"] and not result["error"] and len(result["text"]) > 100:
            results.append(result)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape a URL and print extracted text")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--max-chars", type=int, default=8000, help="Max chars to return")
    args = parser.parse_args()

    result = scrape_url(args.url, max_chars=args.max_chars)
    if result["error"]:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"\n{'='*80}\n")
    print(result["text"])
