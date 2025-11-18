from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


PUBLIC_SCHEMES = {"http", "https"}
LOCAL_HOSTNAMES = {"localhost", "ip6-localhost"}


def is_public_http_url(url: str) -> bool:
    """Return True if the URL is http(s) and host is not private or localhost."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in PUBLIC_SCHEMES:
        return False

    hostname = parsed.hostname or ""
    if not hostname:
        return False

    if hostname.lower() in LOCAL_HOSTNAMES:
        return False

    # If hostname is an IP, validate it is public
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
        return True
    except ValueError:
        # Not an IP, treat as DNS name
        # Block common internal domains
        lowered = hostname.lower()
        if lowered.endswith(".local") or lowered.endswith(".internal") or lowered.endswith(".lan"):
            return False
        return True


def require_public_http_url(url: str) -> None:
    """Raise ValueError if URL is not a public http(s) URL."""
    if not is_public_http_url(url):
        raise ValueError("Blocked URL: only public http(s) URLs are allowed")

