from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Iterable, List
from urllib.parse import urljoin, urlparse

import httpx


async def fetch_text(url: str, timeout: float = 10.0) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.text
    except Exception:
        return None
    return None


async def discover_sitemaps(entry_url: str) -> List[str]:
    """Discover sitemaps from robots.txt or common paths."""
    parsed = urlparse(entry_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    text = await fetch_text(robots_url)
    sitemaps: List[str] = []
    if text:
        for line in text.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(sm)
    # Fallback to common path
    if not sitemaps:
        sitemaps.append(f"{parsed.scheme}://{parsed.netloc}/sitemap.xml")
    return sitemaps


def parse_sitemap_xml(xml_text: str) -> List[str]:
    """Parse sitemap XML and extract URLs."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: List[str] = []
    # urlset
    for url in root.findall(".//sm:url", ns):
        loc = url.find("sm:loc", ns)
        if loc is not None and loc.text:
            urls.append(loc.text.strip())
    # sitemapindex -> nested urlsets (optional)
    for sub in root.findall(".//sm:sitemap", ns):
        loc = sub.find("sm:loc", ns)
        if loc is not None and loc.text:
            urls.append(loc.text.strip())
    return urls


def filter_urls(urls: Iterable[str], include_patterns: List[str], exclude_patterns: List[str]) -> List[str]:
    """Filter URLs based on include/exclude regex patterns."""
    def matches_any(patterns: List[str], text: str) -> bool:
        for p in patterns:
            try:
                if re.search(p, text):
                    return True
            except re.error:
                continue
        return False

    filtered: List[str] = []
    for u in urls:
        if include_patterns and not matches_any(include_patterns, u):
            continue
        if exclude_patterns and matches_any(exclude_patterns, u):
            continue
        filtered.append(u)
    return filtered

