from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic import BaseModel


def generate_run_id(prefix: str | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"{prefix + '_' if prefix else ''}{now}"
    rnd = hashlib.sha1(f"{base}:{now}".encode()).hexdigest()[:6]
    return f"{base}_{rnd}"


def ensure_run_dir(base_output_dir: str, run_id: str) -> Path:
    base = Path(base_output_dir).expanduser().resolve()
    out = base / run_id
    (out / "pages").mkdir(parents=True, exist_ok=True)
    return out


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\-_\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180] if text else "index"


def _build_page_filename(url: str, markdown: str, pages_dir: Path) -> str:
    # docs/openai-agents-like: hostname + path with '/' -> '_', no hashes
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        host = p.netloc or "site"
        path = p.path or "/"
        # drop trailing slash indicator in name by adding trailing underscore if path endswith '/'
        ends_with_slash = path.endswith("/")
        path = path.strip("/")
        path_part = path.replace("/", "_") or "index"
        name_base = f"{host}_{path_part}"
        if ends_with_slash and not name_base.endswith("_"):
            name_base += "_"
    except Exception:
        name_base = "page"
    name_base = _slugify(name_base)
    fname = f"{name_base}.md"
    candidate = pages_dir / fname
    if not candidate.exists():
        return fname
    # Ensure uniqueness with numeric suffix only (no url hash)
    counter = 1
    while True:
        alt = f"{name_base}-{counter}.md"
        candidate = pages_dir / alt
        if not candidate.exists():
            return alt
        counter += 1


def persist_page_markdown(run_dir: Path, url: str, markdown: str) -> Tuple[str, int]:
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    file_name = _build_page_filename(url, markdown or "", pages_dir)
    p = pages_dir / file_name
    data = (markdown or "").encode("utf-8")
    p.write_bytes(data)
    return str(p), len(data)


def append_jsonl(run_dir: Path, record: Dict[str, Any]) -> None:
    out = run_dir / "pages.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_links_csv(run_dir: Path, url: str, links: list[str]) -> None:
    csv_path = run_dir / "links.csv"
    header_needed = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["page_url", "link"])
        for link in links:
            w.writerow([url, link])


def append_log_jsonl(run_dir: Path, event: Dict[str, Any]) -> None:
    log_path = run_dir / "log.ndjson"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


class PageRecord(BaseModel):
    url: str
    status: str  # "ok" or "error"
    path: str | None
    content_bytes: int | None
    error: str | None = None
    duration_ms: int = 0


class Manifest(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    entry: str
    mode: str  # "scrape", "crawl", "site", "sitemap"
    started_at: str
    finished_at: str | None = None
    pages: list[PageRecord] = []
    totals: Dict[str, int] = {"pages_ok": 0, "pages_failed": 0, "bytes_written": 0}
    config: Dict[str, Any] = {}


def write_manifest(run_dir: Path, manifest: Manifest) -> str:
    manifest_path = run_dir / "manifest.json"
    payload = json.loads(manifest.model_dump_json())
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)


def load_manifest(path: str) -> Manifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "pages" in data:
        data["pages"] = [PageRecord(**p) for p in data["pages"]]
    return Manifest(**data)


def update_totals(manifest: Manifest, rec: PageRecord) -> None:
    if rec.status == "ok":
        manifest.totals["pages_ok"] = manifest.totals.get("pages_ok", 0) + 1
        manifest.totals["bytes_written"] = manifest.totals.get("bytes_written", 0) + int(rec.content_bytes or 0)
    else:
        manifest.totals["pages_failed"] = manifest.totals.get("pages_failed", 0) + 1

