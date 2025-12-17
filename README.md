<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling, RAG, and SEO Analysis for AI Agents and AI Coding Assistants</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#tools">Tools</a> •
  <a href="MCP_TOOLS.md">Full API Reference</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling, RAG capabilities, and **comprehensive SEO analysis**.

With this MCP server, you can:
- **Scrape anything** and use that knowledge anywhere for RAG
- **Analyze SEO** with full audits including meta tags, accessibility, readability, and AI-readiness
- **Check AI compatibility** via `llms.txt` and `robots.txt` analysis

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase), and perform RAG over the crawled content.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

2. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system.

3. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

4. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

### Core Crawling
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Safety Guards**: Blocks localhost, private IPs, and internal networks for secure crawling
- **File Persistence**: Optional file-based storage with manifests, JSONL, and CSV exports
- **Adaptive Crawling**: Intelligent crawling that stops when sufficient content is gathered
- **Enhanced Sitemap Discovery**: Automatically discovers sitemaps from robots.txt
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing

### RAG Capabilities
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process
- **Hybrid Search**: Combines keyword and semantic search for better results
- **Code Example Search**: Specialized search for code snippets from documentation

### SEO Analysis
- **Full SEO Audits**: Comprehensive analysis with weighted scoring
- **Meta Tag Analysis**: Title, description, Open Graph, Twitter Cards, JSON-LD
- **Page Structure**: Heading hierarchy, image alt text, internal/external links
- **Accessibility Audit**: ARIA labels, skip links, form labels, lang attribute
- **Readability Analysis**: Flesch-Kincaid scoring for content complexity
- **AI-Readiness Check**: `llms.txt` and `robots.txt` analysis for AI bot access
- **Broken Link Detection**: HTTP status checking for all page links

### Browser Automation
- **Screenshot Capture**: Capture full-page or viewport screenshots as base64 PNG
- **PDF Generation**: Convert any webpage to PDF document
- **JavaScript Execution**: Run custom JS scripts on pages
- **Page Actions**: Click, type, scroll, wait - interact with dynamic content and SPAs

## Tools

The server provides **22 tools** across four categories. For complete API documentation, see **[MCP_TOOLS.md](MCP_TOOLS.md)**.

### Crawling Tools (6 tools)

| Tool | Description | Storage |
|------|-------------|---------|
| `crawl_single_page` | Crawl a single URL | Supabase |
| `crawl_single_page_raw` | Crawl and return content directly | None |
| `smart_crawl_url` | Intelligent crawling (sitemap/page detection) | Supabase |
| `smart_crawl_url_raw` | Smart crawl, return content directly | None |
| `crawl_site` | Full site crawl with disk persistence | Disk |
| `crawl_sitemap` | Crawl all URLs from sitemap.xml | Disk |

### RAG & Search Tools (3 tools)

| Tool | Description | Condition |
|------|-------------|-----------|
| `get_available_sources` | List all indexed sources | Always |
| `perform_rag_query` | Semantic search with optional filtering | Always |
| `search_code_examples` | Search code snippets from docs | `USE_AGENTIC_RAG=true` |

### SEO Analysis Tools (9 tools)

| Tool | Description |
|------|-------------|
| `get_raw_html` | Get raw HTML content (like Firecrawl) |
| `extract_seo_metadata` | Title, meta tags, OG, Twitter Cards, JSON-LD |
| `extract_page_structure` | Headings, images, links analysis |
| `check_broken_links` | HTTP status check for all links |
| `analyze_robots_txt` | Parse robots.txt directives & AI bot access |
| `check_llms_txt` | Check for AI permission files |
| `analyze_accessibility` | ARIA, alt text, skip links, form labels |
| `analyze_readability` | Flesch-Kincaid readability score |
| `full_seo_audit` | **Complete SEO audit** (runs all above in parallel) |

### Browser Automation Tools (4 tools)

| Tool | Description |
|------|-------------|
| `capture_screenshot` | Capture webpage as base64 PNG image |
| `generate_pdf` | Convert webpage to PDF document |
| `execute_javascript` | Run custom JavaScript on pages |
| `crawl_with_actions` | Interact with page (click, type, scroll, wait) then scrape |

### Quick Example: Full SEO Audit

```json
{
  "tool": "full_seo_audit",
  "arguments": {
    "url": "https://example.com"
  }
}
```

**Returns:**
```json
{
  "seo_score": 72,
  "total_issues": 5,
  "component_scores": {
    "meta_tags": 85,
    "structure": 70,
    "robots": 100,
    "llms": 100,
    "accessibility": 55,
    "readability": 45
  },
  "all_issues": [
    "Title too short (35 chars)",
    "12 images missing alt text",
    "No skip-to-content link"
  ]
}
```

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the necessary tables and functions

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# LLM for summaries and contextual embeddings
MODEL_CHOICE=gpt-4.1-nano

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `MODEL_CHOICE`) to generate enriched context that gets embedded alongside the chunk content.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing.

#### 2. **USE_HYBRID_SEARCH**
Combines traditional keyword search with semantic vector search to provide more comprehensive results. The system performs both searches in parallel and intelligently merges results, prioritizing documents that appear in both result sets.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but more robust results, especially for technical content.
- **Cost**: No additional API costs, just computational overhead.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries, and stores them in a separate vector database table specifically designed for code search.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example.
- **Benefits**: Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations.

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.

### Recommended Configurations

**For general documentation RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**For AI coding assistant with code examples:**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
```

**For fast, basic RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

> **Note for Claude Code users**: 
```
claude mcp add-json crawl4ai-rag '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY",
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## New Features & Enhancements

### Safety Guards
All crawling tools now include safety validation that blocks:
- `localhost` and `127.0.0.1`
- Private IP ranges (RFC 1918)
- `file://` schemes
- `.local`, `.internal`, `.lan` domains

This ensures that only public HTTP(S) URLs can be crawled, protecting against SSRF attacks.

### File Persistence
Tools that support file persistence (`crawl_single_page`, `smart_crawl_url`, `crawl_site`, `crawl_sitemap`) can save results to disk with:
- **Manifests**: JSON files with crawl metadata and configuration
- **Markdown files**: Individual page content saved as `.md` files
- **JSONL logs**: Structured logs of all crawled pages
- **CSV links**: Extracted links saved in CSV format

This is especially useful for large crawls where returning all content would bloat the context window.

### Adaptive Crawling
When `adaptive=True` is enabled, the crawler uses intelligent stopping strategies:
- Stops when total content exceeds threshold (default: 5,000 characters)
- Prevents over-crawling for information gathering tasks
- Adjusts threshold based on query complexity

### Enhanced Sitemap Discovery
The `smart_crawl_url` tool now automatically:
- Discovers sitemaps from `robots.txt`
- Falls back to common sitemap paths
- Filters URLs for safety before crawling
- Handles both sitemap.xml and sitemap index files

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers
5. Use the safety, persistence, and adaptive strategy modules for consistent behavior
