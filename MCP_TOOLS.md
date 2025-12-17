# MCP Tools Reference

> Complete reference for all tools available in the Crawl4AI RAG MCP Server

## Table of Contents

- [Core Crawling Tools](#core-crawling-tools)
- [RAG & Search Tools](#rag--search-tools)
- [SEO Analysis Tools](#seo-analysis-tools)
- [Browser Automation Tools](#browser-automation-tools)
- [Tool Categories Overview](#tool-categories-overview)

---

## Core Crawling Tools

### `crawl_single_page`

Crawl a single web page and store its content in Supabase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | The URL to crawl |
| `output_dir` | string | ❌ | null | Directory to save markdown files |

**Returns:** Crawl summary with chunks stored count, or file paths if `output_dir` provided.

**Use Case:** Quick retrieval of a specific page without following links.

```json
{
  "url": "https://example.com/docs/api",
  "output_dir": "./output"
}
```

---

### `crawl_single_page_raw`

Crawl a single web page and return markdown content without storing in Supabase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | The URL to crawl |

**Returns:** Markdown content with metadata (title, word count, links).

**Use Case:** Quick content retrieval without database storage.

---

### `smart_crawl_url`

Intelligently crawl a URL based on its type (sitemap, llms.txt, or webpage) and store in Supabase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | The URL to crawl |
| `max_depth` | int | ❌ | 3 | Maximum recursion depth for link following |
| `max_concurrent` | int | ❌ | 10 | Maximum concurrent crawl operations |
| `chunk_size` | int | ❌ | 5000 | Character limit per chunk |
| `adaptive` | bool | ❌ | false | Enable adaptive stopping (stops when enough content gathered) |
| `output_dir` | string | ❌ | null | Directory to save files |

**URL Type Detection:**
- **Sitemap URLs** (`sitemap.xml`): Extracts and crawls all URLs from sitemap
- **Text files** (`.txt`, `llms.txt`): Parses and follows links
- **Regular pages**: Recursive crawling with link discovery

**Returns:** Comprehensive crawl summary with pages crawled, chunks stored, and sources.

---

### `smart_crawl_url_raw`

Same as `smart_crawl_url` but returns content directly without Supabase storage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | The URL to crawl |
| `max_depth` | int | ❌ | 3 | Maximum recursion depth |
| `max_concurrent` | int | ❌ | 10 | Maximum concurrent operations |

**Returns:** All crawled pages with their markdown content.

---

### `crawl_site`

Comprehensive site crawling with mandatory persistence to disk.

| Parameter | Type | Required | Default | Max | Description |
|-----------|------|----------|---------|-----|-------------|
| `entry_url` | string | ✅ | - | - | Starting URL for the crawl |
| `output_dir` | string | ✅ | - | - | Directory to save all output files |
| `max_depth` | int | ❌ | 2 | 6 | Maximum link depth to follow |
| `max_pages` | int | ❌ | 200 | 5000 | Maximum pages to crawl |
| `adaptive` | bool | ❌ | false | - | Enable adaptive stopping |

**Output Files:**
- `manifest.json` - Crawl metadata and configuration
- `pages/*.md` - Individual page content
- `pages.jsonl` - Structured page records
- `links.csv` - All discovered links

**Returns:** Manifest path and crawl statistics (avoids context bloat).

---

### `crawl_sitemap`

Crawl all URLs from a sitemap.xml with persistence to disk.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sitemap_url` | string | ✅ | - | URL of the sitemap.xml |
| `output_dir` | string | ✅ | - | Directory to save output |
| `max_entries` | int | ❌ | 1000 | Maximum URLs to crawl from sitemap |

**Returns:** Manifest path and crawl statistics.

---

## RAG & Search Tools

### `get_available_sources`

Retrieve all crawled sources (domains) from the database.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| *none* | - | - | No parameters required |

**Returns:** List of sources with:
- Source ID and URL
- Summary description
- Page count and chunk statistics

**Use Case:** Identify available data sources for targeted RAG queries.

---

### `perform_rag_query`

Semantic search over crawled content with optional source filtering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | The search query |
| `source` | string | ❌ | null | Filter by source domain |
| `match_count` | int | ❌ | 5 | Number of results to return |

**Features:**
- Vector similarity search using embeddings
- Optional hybrid search (keyword + semantic) when `USE_HYBRID_SEARCH=true`
- Cross-encoder reranking when `USE_RERANKING=true`

**Returns:** Ranked results with similarity scores and content snippets.

---

### `search_code_examples`

Search specifically for code examples from crawled documentation.

> ⚠️ **Requires:** `USE_AGENTIC_RAG=true` in environment

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Code-related search query |
| `source_id` | string | ❌ | null | Filter by source |
| `match_count` | int | ❌ | 5 | Number of examples to return |

**Returns:** Code blocks with:
- Original code content
- AI-generated summary
- Source URL and context
- Similarity score

**Use Case:** AI coding assistants finding implementation examples.

---

## SEO Analysis Tools

### `get_raw_html`

Retrieve raw HTML content from a webpage (similar to Firecrawl's HTML format).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | URL to fetch |
| `max_length` | int | ❌ | 100000 | Maximum HTML characters to return |

**Returns:**
```json
{
  "success": true,
  "url": "https://example.com",
  "html": "<html>...",
  "html_length": 45000,
  "truncated": false,
  "metadata": {
    "title": "Page Title",
    "description": "Meta description"
  }
}
```

---

### `extract_seo_metadata`

Extract comprehensive SEO metadata from a webpage.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | URL to analyze |

**Extracts:**
- **Title** - Page title with length validation (50-60 chars optimal)
- **Meta tags** - description, keywords, robots, viewport
- **Open Graph** - og:title, og:description, og:image, etc.
- **Twitter Cards** - twitter:card, twitter:title, etc.
- **Canonical URL** - Duplicate content prevention
- **Structured Data** - JSON-LD schemas (Organization, WebPage, Product, etc.)
- **Hreflang tags** - International targeting

**Returns:** Complete metadata object with SEO issue flags.

---

### `extract_page_structure`

Analyze page structure for SEO and accessibility.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | URL to analyze |

**Analyzes:**

**Heading Hierarchy:**
- H1-H6 counts and content
- Validates single H1 rule
- Checks logical hierarchy

**Images:**
- Total count
- Alt text coverage percentage
- Examples of missing alt text

**Links:**
- Internal vs external count
- Anchor text analysis
- Nofollow attributes
- Sample links with destinations

**Returns:** Structural analysis with issue identification.

---

### `check_broken_links`

Check for broken links on a webpage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | Page to check |
| `max_links` | int | ❌ | 50 | Maximum links to validate |

**Checks:**
- HTTP status codes (4xx, 5xx errors)
- Redirects (301, 302)
- Connection timeouts
- SSL certificate issues

**Returns:**
```json
{
  "total_links_checked": 45,
  "broken_links": [...],
  "redirects": [...],
  "healthy_links_count": 42
}
```

---

### `analyze_robots_txt`

Parse and analyze robots.txt directives.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | Domain URL |

**Analyzes:**
- User-agent rules
- Disallow/Allow directives
- Crawl-delay settings
- Sitemap declarations
- AI bot blocking (GPTBot, ChatGPT-User, anthropic-ai, etc.)

**Returns:** Parsed rules with AI accessibility assessment.

---

### `check_llms_txt`

Check for AI-specific permission files.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | Domain URL |

**Checks for:**
- `/llms.txt` - Standard AI permissions file
- `/LLMs.txt` - Case variant
- `/llms-full.txt` - Extended version

**Returns:**
```json
{
  "exists": true,
  "filename": "llms.txt",
  "content_length": 7851,
  "content_preview": "# Site llms.txt\n\n- [Page](url): Description..."
}
```

---

### `analyze_accessibility`

Audit webpage accessibility features.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | URL to audit |

**Checks:**
- `lang` attribute on HTML element
- ARIA labels and roles count
- Image alt text coverage
- Form input labels
- Skip-to-content links
- Heading structure

**Returns:** Accessibility score (0-100) with specific issues.

---

### `analyze_readability`

Calculate content readability metrics.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | URL to analyze |

**Metrics:**
- **Flesch Reading Ease** (0-100, higher = easier)
  - 90-100: Very Easy (5th grade)
  - 60-70: Standard (8th-9th grade)
  - 30-50: Difficult (College)
  - 0-30: Very Difficult (Professional)
- Average words per sentence
- Average syllables per word
- Total word/sentence counts

**Returns:**
```json
{
  "flesch_reading_ease": 45.2,
  "readability_level": "Difficult",
  "metrics": {
    "total_words": 1500,
    "total_sentences": 85,
    "avg_words_per_sentence": 17.6
  }
}
```

---

### `full_seo_audit`

Comprehensive SEO audit combining all analysis tools.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | ✅ | URL to audit |

**Runs in Parallel:**
1. `extract_seo_metadata`
2. `extract_page_structure`
3. `analyze_robots_txt`
4. `check_llms_txt`
5. `analyze_accessibility`
6. `analyze_readability`

**Scoring (Weighted):**
| Component | Weight |
|-----------|--------|
| Meta Tags | 25% |
| Page Structure | 20% |
| Robots.txt | 10% |
| llms.txt | 15% |
| Accessibility | 15% |
| Readability | 15% |

**Returns:**
```json
{
  "success": true,
  "url": "https://example.com",
  "seo_score": 72,
  "total_issues": 5,
  "all_issues": [
    "Title too short (35 chars)",
    "12 images missing alt text",
    "No skip-to-content link"
  ],
  "component_scores": {
    "meta_tags": 85,
    "structure": 70,
    "robots": 100,
    "llms": 100,
    "accessibility": 55,
    "readability": 45
  },
  "metadata": {...},
  "structure": {...},
  "robots": {...},
  "llms_txt": {...},
  "accessibility": {...},
  "readability": {...}
}
```

---

## Browser Automation Tools

### `capture_screenshot`

Capture a screenshot of a webpage as base64 PNG.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | URL of the webpage to screenshot |
| `full_page` | bool | ❌ | true | Capture full scrollable page vs viewport only |

**Use Cases:**
- Visual verification of page rendering
- AI vision analysis of webpage layout
- Capturing dynamic/JS-rendered content
- Documentation and archiving

**Returns:**
```json
{
  "success": true,
  "url": "https://example.com",
  "screenshot": "iVBORw0KGgoAAAANSUhEUgAA...",
  "format": "base64_png",
  "full_page": true,
  "message": "Screenshot captured successfully. Use base64 decode to get PNG image."
}
```

---

### `generate_pdf`

Generate a PDF document from a webpage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | URL of the webpage to convert |

**Use Cases:**
- Archiving web pages as documents
- Creating printable versions of web content
- Capturing very long pages that exceed screenshot limits
- Document processing pipelines

**Returns:**
```json
{
  "success": true,
  "url": "https://example.com",
  "pdf": "JVBERi0xLjQKJeLjz9MKNSAw...",
  "format": "base64_pdf",
  "message": "PDF generated successfully. Use base64 decode to get PDF file."
}
```

---

### `execute_javascript`

Execute custom JavaScript on a webpage and return results.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | URL of the webpage |
| `scripts` | list[str] | ✅ | - | JavaScript code snippets to execute |

**Use Cases:**
- Extracting data from JavaScript-rendered pages
- Interacting with dynamic content
- Testing page behavior
- Scraping SPAs (Single Page Applications)

**Example Scripts:**
```javascript
// Get page title
"return document.title"

// Get all links
"return Array.from(document.querySelectorAll('a')).map(a => a.href)"

// Click a button
"document.querySelector('button.load-more').click()"

// Scroll to bottom
"window.scrollTo(0, document.body.scrollHeight)"
```

**Returns:**
```json
{
  "success": true,
  "url": "https://example.com",
  "scripts_executed": 3,
  "markdown": "# Page Content After JS Execution...",
  "html_length": 45000,
  "message": "Successfully executed 3 script(s) on the page."
}
```

---

### `crawl_with_actions`

Crawl a webpage after performing browser actions (click, type, scroll, wait).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | ✅ | - | URL of the webpage |
| `actions` | list[dict] | ✅ | - | List of actions to perform |
| `screenshot` | bool | ❌ | false | Capture screenshot after actions |

**Action Types:**

| Action | Parameters | Description |
|--------|------------|-------------|
| `click` | `selector` | Click an element |
| `type` | `selector`, `text` | Type text into input |
| `scroll` | `direction`, `amount` | Scroll the page |
| `wait` | `milliseconds` | Wait for time |
| `wait_for` | `selector`, `timeout` | Wait for element |
| `press` | `key`, `selector` (optional) | Press a key |

**Example:**
```json
{
  "url": "https://example.com/search",
  "actions": [
    {"type": "click", "selector": "button.accept-cookies"},
    {"type": "wait", "milliseconds": 500},
    {"type": "type", "selector": "input#search", "text": "AI tools"},
    {"type": "press", "selector": "input#search", "key": "Enter"},
    {"type": "wait_for", "selector": ".search-results", "timeout": 5000},
    {"type": "scroll", "direction": "down", "amount": 1000}
  ],
  "screenshot": true
}
```

**Scroll Directions:**
- `"down"` - Scroll down by `amount` pixels
- `"up"` - Scroll up by `amount` pixels
- `"bottom"` - Scroll to page bottom
- `"top"` - Scroll to page top

**Returns:**
```json
{
  "success": true,
  "url": "https://example.com/search",
  "actions_performed": 6,
  "markdown": "# Search Results...",
  "html_length": 35000,
  "screenshot": "iVBORw0KGgoAAAANSUhEUgAA...",
  "screenshot_format": "base64_png"
}
```

---

## Tool Categories Overview

### By Function

| Category | Tools | Description |
|----------|-------|-------------|
| **Crawling** | `crawl_single_page`, `crawl_single_page_raw`, `smart_crawl_url`, `smart_crawl_url_raw`, `crawl_site`, `crawl_sitemap` | Web content extraction |
| **RAG** | `get_available_sources`, `perform_rag_query`, `search_code_examples` | Vector search and retrieval |
| **SEO** | `get_raw_html`, `extract_seo_metadata`, `extract_page_structure`, `check_broken_links`, `analyze_robots_txt`, `check_llms_txt`, `analyze_accessibility`, `analyze_readability`, `full_seo_audit` | SEO analysis and auditing |
| **Browser Automation** | `capture_screenshot`, `generate_pdf`, `execute_javascript`, `crawl_with_actions` | Browser control and page interaction |

### By Storage Behavior

| Behavior | Tools |
|----------|-------|
| **Stores in Supabase** | `crawl_single_page`, `smart_crawl_url` |
| **Returns content directly** | `crawl_single_page_raw`, `smart_crawl_url_raw`, all SEO tools, all Browser Automation tools |
| **Saves to disk** | `crawl_site`, `crawl_sitemap` |

### By Conditional Availability

| Condition | Tool |
|-----------|------|
| `USE_AGENTIC_RAG=true` | `search_code_examples` |
| `USE_RERANKING=true` | Reranking applied to `perform_rag_query` results |
| `USE_HYBRID_SEARCH=true` | Hybrid search in `perform_rag_query` |

---

## Safety Features

All crawling tools include safety validation:

- ❌ Blocks `localhost` and `127.0.0.1`
- ❌ Blocks private IP ranges (RFC 1918)
- ❌ Blocks `file://` schemes
- ❌ Blocks `.local`, `.internal`, `.lan` domains
- ✅ Only allows public HTTP(S) URLs

---

## Example Workflows

### 1. Quick SEO Audit

```python
# Single comprehensive audit
result = await full_seo_audit(url="https://example.com")
```

### 2. Documentation Indexing

```python
# Index documentation site
await smart_crawl_url(
    url="https://docs.example.com",
    max_depth=3,
    adaptive=True
)

# Query indexed content
results = await perform_rag_query(
    query="How to authenticate?",
    source="docs.example.com"
)
```

### 3. Large Site Crawl

```python
# Persist to disk for large sites
await crawl_site(
    entry_url="https://example.com",
    output_dir="./crawl_output",
    max_pages=1000,
    max_depth=4
)
```

### 4. AI-Readiness Check

```python
# Check AI-specific files
llms = await check_llms_txt(url="https://example.com")
robots = await analyze_robots_txt(url="https://example.com")
```

### 5. Scrape Dynamic SPA Content

```python
# Navigate and interact with a single-page app
result = await crawl_with_actions(
    url="https://spa-example.com",
    actions=[
        {"type": "wait", "milliseconds": 2000},
        {"type": "click", "selector": "button.load-data"},
        {"type": "wait_for", "selector": ".data-loaded"},
        {"type": "scroll", "direction": "bottom"}
    ],
    screenshot=True
)
```

### 6. Generate Documentation Archive

```python
# Create PDF of documentation page
pdf_result = await generate_pdf(url="https://docs.example.com/api")

# Save to file
import base64
with open("api_docs.pdf", "wb") as f:
    f.write(base64.b64decode(pdf_result["pdf"]))
```
