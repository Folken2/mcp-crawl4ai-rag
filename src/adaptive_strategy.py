from __future__ import annotations

from typing import List


def should_continue_crawling(
    pages_so_far: List[str],  # markdown content of pages crawled so far
    max_pages: int,
    target_content_threshold: int = 5000,  # characters
) -> bool:
    """
    Simple adaptive strategy: stop crawling if we have enough content
    or hit the page limit.
    
    A more sophisticated version could use LLM-based information foraging
    to determine if the query has been sufficiently answered.
    """
    if len(pages_so_far) >= max_pages:
        return False
    
    # Calculate total content gathered
    total_content_chars = sum(len(content) for content in pages_so_far)
    
    # Stop if we have gathered enough content
    if total_content_chars >= target_content_threshold:
        return False
    
    return True


def get_adaptive_threshold(query: str | None = None) -> int:
    """
    Get content threshold based on query complexity.
    More complex queries might need more content.
    """
    if not query:
        return 5000
    
    # Simple heuristic: longer/more complex queries need more content
    if len(query) > 100 or "detailed" in query.lower() or "comprehensive" in query.lower():
        return 8000
    elif len(query) < 30:
        return 3000
    else:
        return 5000

