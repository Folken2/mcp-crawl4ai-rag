"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import concurrent.futures

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

from utils import (
    get_supabase_client, 
    add_documents_to_supabase, 
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    search_code_examples
)
from safety import require_public_http_url, is_public_http_url
from persistence import (
    generate_run_id,
    ensure_run_dir,
    persist_page_markdown,
    append_links_csv,
    Manifest,
    PageRecord,
    write_manifest,
    update_totals
)
from adaptive_strategy import should_continue_crawling
from sitemap_utils import discover_sitemaps, parse_sitemap_xml, filter_urls, fetch_text

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Optional[Client]
    reranking_model: Optional[CrossEncoder] = None

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Supabase client
    supabase_client = get_supabase_client()
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model
        )
    finally:
        # Clean up all components
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (code, context_before, context_after)
        
    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str, output_dir: Optional[str] = None) -> str:
    """
    Crawl a single web page and store its content in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
        output_dir: Optional directory to persist content to disk. If provided, content is saved to files
                    and metadata is returned instead of full content (avoids context bloat).
    
    Returns:
        Summary of the crawling operation and storage in Supabase (or file persistence if output_dir provided)
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({
                "success": False,
                "url": url,
                "error": "URL is required and cannot be empty"
            }, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Invalid URL format: '{url}'. URL must start with 'http://' or 'https://'"
            }, indent=2)
        
        # Safety check: ensure URL is public
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)
        
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if Supabase is configured
        if not supabase_client:
            return json.dumps({
                "success": False,
                "url": url,
                "error": (
                    "Supabase is not configured. This tool requires Supabase to store content. "
                    "Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file. "
                    "Alternatively, use 'crawl_single_page_raw' to get content without storing it."
                )
            }, indent=2)
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Handle file persistence if output_dir is provided
            if output_dir:
                from datetime import datetime, timezone
                run_id = generate_run_id("scrape")
                run_dir = ensure_run_dir(output_dir, run_id)
                started_at = datetime.now(timezone.utc)
                
                # Create manifest
                manifest = Manifest(
                    run_id=run_id,
                    entry=url,
                    mode="scrape",
                    started_at=started_at.isoformat(),
                    config={"url": url}
                )
                write_manifest(run_dir, manifest)
                
                # Persist page to disk
                file_path, content_bytes = persist_page_markdown(run_dir, url, result.markdown)
                
                # Extract links
                links = []
                raw_links = getattr(result, "links", {}) or {}
                for link_list in [raw_links.get("internal", []), raw_links.get("external", [])]:
                    for link in link_list:
                        if isinstance(link, str):
                            links.append(link)
                        else:
                            href = link.get("href") or link.get("url")
                            if isinstance(href, str):
                                links.append(href)
                
                if links:
                    append_links_csv(run_dir, url, links)
                
                # Create page record
                page_record = PageRecord(
                    url=url,
                    status="ok",
                    error=None,
                    duration_ms=0,
                    path=file_path,
                    content_bytes=content_bytes
                )
                manifest.pages = [page_record]
                manifest.finished_at = datetime.now(timezone.utc).isoformat()
                manifest.totals = {"pages_ok": 1, "pages_failed": 0, "bytes_written": content_bytes}
                write_manifest(run_dir, manifest)
                
                return json.dumps({
                    "success": True,
                    "run_id": run_id,
                    "output_dir": str(run_dir),
                    "manifest_path": str(run_dir / "manifest.json"),
                    "file_path": file_path,
                    "bytes_written": content_bytes,
                    "started_at": started_at.isoformat(),
                    "finished_at": manifest.finished_at
                }, indent=2)
            
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                total_word_count += meta.get("word_count", 0)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Update source information FIRST (before inserting documents)
            try:
                source_summary = extract_source_summary(source_id, result.markdown[:5000])  # Use first 5000 chars for summary
                update_source_info(supabase_client, source_id, source_summary, total_word_count)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Failed to update source information in Supabase: {str(e)}. Please check your Supabase connection and credentials."
                }, indent=2)
            
            # Add documentation chunks to Supabase (AFTER source exists)
            try:
                add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)
            except ValueError as e:
                # ValueError from our utility functions contains user-friendly messages
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": str(e)
                }, indent=2)
            except Exception as e:
                error_msg = str(e).lower()
                if 'connection' in error_msg or 'network' in error_msg:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": f"Failed to connect to Supabase: {str(e)}. Please check your network connection and SUPABASE_URL."
                    }, indent=2)
                elif 'permission' in error_msg or 'unauthorized' in error_msg or '401' in error_msg:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": f"Supabase authentication failed: {str(e)}. Please verify your SUPABASE_SERVICE_KEY is correct."
                    }, indent=2)
                else:
                    return json.dumps({
                        "success": False,
                        "url": url,
                        "error": f"Failed to store content in Supabase: {str(e)}. Please check your Supabase configuration."
                    }, indent=2)
            
            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            if extract_code_examples:
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
                    
                    # Add code examples to Supabase
                    try:
                        add_code_examples_to_supabase(
                            supabase_client, 
                            code_urls, 
                            code_chunk_numbers, 
                            code_examples, 
                            code_summaries, 
                            code_metadatas
                        )
                    except Exception as e:
                        # Log error but don't fail the entire operation
                        print(f"Warning: Failed to store code examples in Supabase: {e}")
                        # Continue with the response - main content was stored successfully
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "code_examples_stored": len(code_blocks) if code_blocks else 0,
                "content_length": len(result.markdown),
                "total_word_count": total_word_count,
                "source_id": source_id,
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawling error"
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Failed to crawl URL: {error_msg}"
            }, indent=2)
    except ValueError as e:
        # ValueError from our utility functions contains user-friendly messages
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)
    except Exception as e:
        error_msg = str(e).lower()
        if 'timeout' in error_msg or 'connection' in error_msg:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Network error while crawling: {str(e)}. Please check your internet connection and try again."
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Unexpected error: {str(e)}"
            }, indent=2)

@mcp.tool()
async def crawl_single_page_raw(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and return its markdown content without storing in Supabase.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links
    and without requiring Supabase configuration. Perfect for agents that just need to scrape content.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        JSON string with the crawled content in markdown format
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({
                "success": False,
                "url": url,
                "error": "URL is required and cannot be empty"
            }, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Invalid URL format: '{url}'. URL must start with 'http://' or 'https://'"
            }, indent=2)
        
        # Safety check: ensure URL is public
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)
        
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            return json.dumps({
                "success": True,
                "url": url,
                "markdown": result.markdown,
                "content_length": len(result.markdown),
                "word_count": len(result.markdown.split()),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                },
                "internal_links": [link.get("href", "") for link in result.links.get("internal", [])[:10]],
                "external_links": [link.get("href", "") for link in result.links.get("external", [])[:10]]
            }, indent=2)
        else:
            error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawling error"
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Failed to crawl URL: {error_msg}"
            }, indent=2)
    except Exception as e:
        error_msg = str(e).lower()
        if 'timeout' in error_msg or 'connection' in error_msg:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Network error while crawling: {str(e)}. Please check your internet connection and try again."
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Unexpected error: {str(e)}"
            }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000, adaptive: bool = False, output_dir: Optional[str] = None) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel (also discovers sitemaps from robots.txt)
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
        adaptive: Enable adaptive crawling to stop when sufficient content is gathered (default: False)
        output_dir: Optional directory to persist content to disk. If provided, content is saved to files
                    and metadata is returned instead of full content (avoids context bloat).
    
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({
                "success": False,
                "url": url,
                "error": "URL is required and cannot be empty"
            }, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Invalid URL format: '{url}'. URL must start with 'http://' or 'https://'"
            }, indent=2)
        
        # Safety check: ensure URL is public
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)
        
        # Validate parameters
        if max_depth < 1 or max_depth > 10:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"max_depth must be between 1 and 10, got {max_depth}"
            }, indent=2)
        
        if max_concurrent < 1 or max_concurrent > 50:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"max_concurrent must be between 1 and 50, got {max_concurrent}"
            }, indent=2)
        
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Check if Supabase is configured
        if not supabase_client:
            return json.dumps({
                "success": False,
                "url": url,
                "error": (
                    "Supabase is not configured. This tool requires Supabase to store content. "
                    "Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file. "
                    "Alternatively, use 'smart_crawl_url_raw' to get content without storing it."
                )
            }, indent=2)
        
        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            # Filter URLs for safety
            safe_urls = [u for u in sitemap_urls if is_public_http_url(u)]
            if not safe_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No safe URLs found in sitemap after filtering"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, safe_urls, max_concurrent=max_concurrent, adaptive=adaptive)
            crawl_type = "sitemap"
        else:
            # Try to discover sitemaps from robots.txt first
            discovered_sitemaps = await discover_sitemaps(url)
            if discovered_sitemaps:
                # Use discovered sitemap if available
                sitemap_url = discovered_sitemaps[0]
                sitemap_text = await fetch_text(sitemap_url)
                if sitemap_text:
                    sitemap_urls = parse_sitemap_xml(sitemap_text)
                    safe_urls = [u for u in sitemap_urls if is_public_http_url(u)]
                    if safe_urls:
                        crawl_results = await crawl_batch(crawler, safe_urls, max_concurrent=max_concurrent, adaptive=adaptive)
                        crawl_type = "sitemap_discovered"
            
            # If no sitemap or sitemap crawl failed, use recursive crawl
            if not crawl_results:
                crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent, adaptive=adaptive)
                crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Handle file persistence if output_dir is provided
        if output_dir:
            from datetime import datetime, timezone
            run_id = generate_run_id("crawl")
            run_dir = ensure_run_dir(output_dir, run_id)
            started_at = datetime.now(timezone.utc)
            
            # Create manifest
            manifest = Manifest(
                run_id=run_id,
                entry=url,
                mode="crawl",
                started_at=started_at.isoformat(),
                config={
                    "url": url,
                    "max_depth": max_depth,
                    "max_concurrent": max_concurrent,
                    "adaptive": adaptive,
                    "crawl_type": crawl_type
                }
            )
            write_manifest(run_dir, manifest)
            
            pages_ok = 0
            pages_failed = 0
            total_bytes = 0
            
            # Persist each page
            for doc in crawl_results:
                try:
                    file_path, content_bytes = persist_page_markdown(run_dir, doc['url'], doc['markdown'])
                    
                    # Extract links
                    links = []
                    # Links might be in the doc if we stored them, otherwise empty
                    if links:
                        append_links_csv(run_dir, doc['url'], links)
                    
                    page_record = PageRecord(
                        url=doc['url'],
                        status="ok",
                        error=None,
                        duration_ms=0,
                        path=file_path,
                        content_bytes=content_bytes
                    )
                    manifest.pages.append(page_record)
                    update_totals(manifest, page_record)
                    pages_ok += 1
                    total_bytes += content_bytes
                except Exception as e:
                    page_record = PageRecord(
                        url=doc['url'],
                        status="failed",
                        error=str(e),
                        duration_ms=0,
                        path=None,
                        content_bytes=None
                    )
                    manifest.pages.append(page_record)
                    update_totals(manifest, page_record)
                    pages_failed += 1
            
            manifest.finished_at = datetime.now(timezone.utc).isoformat()
            write_manifest(run_dir, manifest)
            
            return json.dumps({
                "success": True,
                "run_id": run_id,
                "output_dir": str(run_dir),
                "manifest_path": str(run_dir / "manifest.json"),
                "pages_ok": pages_ok,
                "pages_failed": pages_failed,
                "bytes_written": total_bytes,
                "started_at": started_at.isoformat(),
                "finished_at": manifest.finished_at,
                "crawl_type": crawl_type
            }, indent=2)
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}
        
        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)
                
                chunk_count += 1
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']
        
        # Update source information for each unique source FIRST (before inserting documents)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
                source_summaries = list(executor.map(lambda args: extract_source_summary(args[0], args[1]), source_summary_args))
            
            for (source_id, _), summary in zip(source_summary_args, source_summaries):
                word_count = source_word_counts.get(source_id, 0)
                update_source_info(supabase_client, source_id, summary, word_count)
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Failed to update source information in Supabase: {str(e)}. Please check your Supabase connection and credentials."
            }, indent=2)
        
        # Add documentation chunks to Supabase (AFTER sources exist)
        try:
            batch_size = 20
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)
        except ValueError as e:
            # ValueError from our utility functions contains user-friendly messages
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)
        except Exception as e:
            error_msg = str(e).lower()
            if 'connection' in error_msg or 'network' in error_msg:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Failed to connect to Supabase: {str(e)}. Please check your network connection and SUPABASE_URL."
                }, indent=2)
            elif 'permission' in error_msg or 'unauthorized' in error_msg or '401' in error_msg:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Supabase authentication failed: {str(e)}. Please verify your SUPABASE_SERVICE_KEY is correct."
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": f"Failed to store content in Supabase: {str(e)}. Please check your Supabase configuration."
                }, indent=2)
        
        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if extract_code_examples_enabled:
            all_code_blocks = []
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                code_blocks = extract_code_blocks(md)
                
                if code_blocks:
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [(block['code'], block['context_before'], block['context_after']) 
                                        for block in code_blocks]
                        
                        # Generate summaries in parallel
                        summaries = list(executor.map(process_code_example, summary_args))
                    
                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))  # Use global code example index
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
            
            # Add all code examples to Supabase
            if code_examples:
                try:
                    add_code_examples_to_supabase(
                        supabase_client, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas,
                        batch_size=batch_size
                    )
                except Exception as e:
                    # Log error but don't fail the entire operation
                    print(f"Warning: Failed to store code examples in Supabase: {e}")
                    # Continue with the response - main content was stored successfully
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": len(code_examples) if 'code_examples' in locals() else 0,
            "sources_updated": len(source_content_map),
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except ValueError as e:
        # ValueError from our utility functions contains user-friendly messages
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)
    except Exception as e:
        error_msg = str(e).lower()
        if 'timeout' in error_msg or 'connection' in error_msg:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Network error while crawling: {str(e)}. Please check your internet connection and try again."
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Unexpected error: {str(e)}"
            }, indent=2)

@mcp.tool()
async def smart_crawl_url_raw(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10) -> str:
    """
    Intelligently crawl a URL based on its type and return markdown content without storing in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is returned as markdown without requiring Supabase configuration.
    Perfect for agents that just need to scrape content without storage.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
    
    Returns:
        JSON string with all crawled pages and their markdown content
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({
                "success": False,
                "url": url,
                "error": "URL is required and cannot be empty"
            }, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Invalid URL format: '{url}'. URL must start with 'http://' or 'https://'"
            }, indent=2)
        
        # Validate parameters
        if max_depth < 1 or max_depth > 10:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"max_depth must be between 1 and 10, got {max_depth}"
            }, indent=2)
        
        if max_concurrent < 1 or max_concurrent > 50:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"max_concurrent must be between 1 and 50, got {max_concurrent}"
            }, indent=2)
        
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Format results with markdown content
        pages = []
        total_content_length = 0
        
        for doc in crawl_results:
            content_length = len(doc['markdown'])
            total_content_length += content_length
            pages.append({
                "url": doc['url'],
                "markdown": doc['markdown'],
                "content_length": content_length,
                "word_count": len(doc['markdown'].split())
            })
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "total_content_length": total_content_length,
            "pages": pages
        }, indent=2)
    except ValueError as e:
        # ValueError from our utility functions contains user-friendly messages
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)
    except Exception as e:
        error_msg = str(e).lower()
        if 'timeout' in error_msg or 'connection' in error_msg:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Network error while crawling: {str(e)}. Please check your internet connection and try again."
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Unexpected error: {str(e)}"
            }, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering 
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        if not supabase_client:
            return json.dumps({
                "success": False,
                "error": "Supabase is not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
            }, indent=2)
        
        # Query the sources table directly
        result = supabase_client.from_('sources')\
            .select('*')\
            .order('source_id')\
            .execute()
        
        # Format the sources with their details
        sources = []
        if result.data:
            for source in result.data:
                sources.append({
                    "source_id": source.get("source_id"),
                    "summary": source.get("summary"),
                    "total_words": source.get("total_words"),
                    "created_at": source.get("created_at"),
                    "updated_at": source.get("updated_at")
                })
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def crawl_site(ctx: Context, entry_url: str, output_dir: str, max_depth: int = 2, max_pages: int = 200, adaptive: bool = False) -> str:
    """
    Comprehensive site crawling with persistence to disk.
    
    This tool crawls an entire website and persists all results to disk. It always requires
    an output_dir and returns metadata only (avoids context bloat).
    
    Args:
        ctx: The MCP server provided context
        entry_url: Starting URL for site crawl
        output_dir: Directory to persist results (required)
        max_depth: Maximum crawl depth (default: 2, max: 6)
        max_pages: Maximum pages to crawl (default: 200, max: 5000)
        adaptive: Enable adaptive crawling to stop when sufficient content is gathered (default: False)
    
    Returns:
        JSON string with crawl summary and file paths
    """
    try:
        # Validate URL
        if not entry_url or not entry_url.strip():
            return json.dumps({
                "success": False,
                "url": entry_url,
                "error": "URL is required and cannot be empty"
            }, indent=2)
        
        if not entry_url.startswith(('http://', 'https://')):
            return json.dumps({
                "success": False,
                "url": entry_url,
                "error": f"Invalid URL format: '{entry_url}'. URL must start with 'http://' or 'https://'"
            }, indent=2)
        
        # Safety check
        try:
            require_public_http_url(entry_url)
        except ValueError as e:
            return json.dumps({
                "success": False,
                "url": entry_url,
                "error": str(e)
            }, indent=2)
        
        # Validate parameters
        if max_depth < 1 or max_depth > 6:
            return json.dumps({
                "success": False,
                "url": entry_url,
                "error": f"max_depth must be between 1 and 6, got {max_depth}"
            }, indent=2)
        
        if max_pages < 1 or max_pages > 5000:
            return json.dumps({
                "success": False,
                "url": entry_url,
                "error": f"max_pages must be between 1 and 5000, got {max_pages}"
            }, indent=2)
        
        # Use smart_crawl_url with output_dir (which will handle persistence)
        # But we need to limit pages, so we'll do a custom crawl
        crawler = ctx.request_context.lifespan_context.crawler
        
        from datetime import datetime, timezone
        run_id = generate_run_id("site")
        run_dir = ensure_run_dir(output_dir, run_id)
        started_at = datetime.now(timezone.utc)
        
        # Create manifest
        manifest = Manifest(
            run_id=run_id,
            entry=entry_url,
            mode="site",
            started_at=started_at.isoformat(),
            config={
                "entry_url": entry_url,
                "max_depth": max_depth,
                "max_pages": max_pages,
                "adaptive": adaptive
            }
        )
        write_manifest(run_dir, manifest)
        
        # Perform recursive crawl with page limit
        crawl_results = await crawl_recursive_internal_links(
            crawler, 
            [entry_url], 
            max_depth=max_depth, 
            max_concurrent=10,
            adaptive=adaptive
        )
        
        # Limit to max_pages
        crawl_results = crawl_results[:max_pages]
        
        pages_ok = 0
        pages_failed = 0
        total_bytes = 0
        
        # Persist each page
        for doc in crawl_results:
            try:
                file_path, content_bytes = persist_page_markdown(run_dir, doc['url'], doc['markdown'])
                
                page_record = PageRecord(
                    url=doc['url'],
                    status="ok",
                    error=None,
                    duration_ms=0,
                    path=file_path,
                    content_bytes=content_bytes
                )
                manifest.pages.append(page_record)
                update_totals(manifest, page_record)
                pages_ok += 1
                total_bytes += content_bytes
            except Exception as e:
                page_record = PageRecord(
                    url=doc['url'],
                    status="failed",
                    error=str(e),
                    duration_ms=0,
                    path=None,
                    content_bytes=None
                )
                manifest.pages.append(page_record)
                update_totals(manifest, page_record)
                pages_failed += 1
        
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        
        return json.dumps({
            "success": True,
            "run_id": run_id,
            "output_dir": str(run_dir),
            "manifest_path": str(run_dir / "manifest.json"),
            "pages_ok": pages_ok,
            "pages_failed": pages_failed,
            "bytes_written": total_bytes,
            "started_at": started_at.isoformat(),
            "finished_at": manifest.finished_at
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": entry_url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def crawl_sitemap(ctx: Context, sitemap_url: str, output_dir: str, max_entries: int = 1000) -> str:
    """
    Crawl URLs discovered from sitemap.xml with persistence to disk.
    
    This tool fetches a sitemap, extracts URLs, and crawls them all while persisting
    results to disk. It always requires an output_dir and returns metadata only.
    
    Args:
        ctx: The MCP server provided context
        sitemap_url: URL to sitemap.xml
        output_dir: Directory to persist results (required)
        max_entries: Maximum sitemap entries to process (default: 1000)
    
    Returns:
        JSON string with crawl summary and file paths
    """
    try:
        # Validate URL
        if not sitemap_url or not sitemap_url.strip():
            return json.dumps({
                "success": False,
                "url": sitemap_url,
                "error": "URL is required and cannot be empty"
            }, indent=2)
        
        if not sitemap_url.startswith(('http://', 'https://')):
            return json.dumps({
                "success": False,
                "url": sitemap_url,
                "error": f"Invalid URL format: '{sitemap_url}'. URL must start with 'http://' or 'https://'"
            }, indent=2)
        
        # Safety check
        try:
            require_public_http_url(sitemap_url)
        except ValueError as e:
            return json.dumps({
                "success": False,
                "url": sitemap_url,
                "error": str(e)
            }, indent=2)
        
        # Fetch and parse sitemap
        sitemap_text = await fetch_text(sitemap_url)
        if not sitemap_text:
            return json.dumps({
                "success": False,
                "url": sitemap_url,
                "error": "Failed to fetch sitemap"
            }, indent=2)
        
        sitemap_urls = parse_sitemap_xml(sitemap_text)
        if not sitemap_urls:
            return json.dumps({
                "success": False,
                "url": sitemap_url,
                "error": "No URLs found in sitemap"
            }, indent=2)
        
        # Filter for safety and limit entries
        safe_urls = [u for u in sitemap_urls if is_public_http_url(u)][:max_entries]
        if not safe_urls:
            return json.dumps({
                "success": False,
                "url": sitemap_url,
                "error": "No safe URLs found in sitemap after filtering"
            }, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        
        from datetime import datetime, timezone
        run_id = generate_run_id("sitemap")
        run_dir = ensure_run_dir(output_dir, run_id)
        started_at = datetime.now(timezone.utc)
        
        # Create manifest
        manifest = Manifest(
            run_id=run_id,
            entry=sitemap_url,
            mode="sitemap",
            started_at=started_at.isoformat(),
            config={
                "sitemap_url": sitemap_url,
                "max_entries": max_entries
            }
        )
        write_manifest(run_dir, manifest)
        
        # Crawl all URLs
        crawl_results = await crawl_batch(crawler, safe_urls, max_concurrent=10, adaptive=False)
        
        pages_ok = 0
        pages_failed = 0
        total_bytes = 0
        
        # Persist each page
        for doc in crawl_results:
            try:
                file_path, content_bytes = persist_page_markdown(run_dir, doc['url'], doc['markdown'])
                
                page_record = PageRecord(
                    url=doc['url'],
                    status="ok",
                    error=None,
                    duration_ms=0,
                    path=file_path,
                    content_bytes=content_bytes
                )
                manifest.pages.append(page_record)
                update_totals(manifest, page_record)
                pages_ok += 1
                total_bytes += content_bytes
            except Exception as e:
                page_record = PageRecord(
                    url=doc['url'],
                    status="failed",
                    error=str(e),
                    duration_ms=0,
                    path=None,
                    content_bytes=None
                )
                manifest.pages.append(page_record)
                update_totals(manifest, page_record)
                pages_failed += 1
        
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        write_manifest(run_dir, manifest)
        
        return json.dumps({
            "success": True,
            "run_id": run_id,
            "output_dir": str(run_dir),
            "manifest_path": str(run_dir / "manifest.json"),
            "pages_ok": pages_ok,
            "pages_failed": pages_failed,
            "bytes_written": total_bytes,
            "started_at": started_at.isoformat(),
            "finished_at": manifest.finished_at
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": sitemap_url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    try:
        # Validate query
        if not query or not query.strip():
            return json.dumps({
                "success": False,
                "query": query,
                "error": "Query is required and cannot be empty"
            }, indent=2)
        
        # Validate match_count
        if match_count < 1 or match_count > 100:
            return json.dumps({
                "success": False,
                "query": query,
                "error": f"match_count must be between 1 and 100, got {match_count}"
            }, indent=2)
        
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        if not supabase_client:
            return json.dumps({
                "success": False,
                "query": query,
                "error": (
                    "Supabase is not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables. "
                    "You need to crawl and store content first using 'crawl_single_page' or 'smart_crawl_url' before querying."
                )
            }, indent=2)
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE
            keyword_query = supabase_client.from_('crawled_pages')\
                .select('id, url, chunk_number, content, metadata, source_id')\
                .ilike('content', f'%{query}%')
            
            # Apply source filter if provided
            if source and source.strip():
                keyword_query = keyword_query.eq('source_id', source)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: int = 5) -> str:
    """
    Search for code examples relevant to the query.
    
    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps({
            "success": False,
            "query": query,
            "error": (
                "Code example extraction is disabled. Set USE_AGENTIC_RAG=true in your .env file to enable this feature. "
                "Alternatively, use 'perform_rag_query' for regular content search."
            )
        }, indent=2)
    
    try:
        # Validate query
        if not query or not query.strip():
            return json.dumps({
                "success": False,
                "query": query,
                "error": "Query is required and cannot be empty"
            }, indent=2)
        
        # Validate match_count
        if match_count < 1 or match_count > 100:
            return json.dumps({
                "success": False,
                "query": query,
                "error": f"match_count must be between 1 and 100, got {match_count}"
            }, indent=2)
        
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        if not supabase_client:
            return json.dumps({
                "success": False,
                "query": query,
                "error": (
                    "Supabase is not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables. "
                    "You need to crawl and store content with USE_AGENTIC_RAG=true first before searching code examples."
                )
            }, indent=2)
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # Import the search function from utils
            from utils import search_code_examples as search_code_examples_impl
            
            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE on both content and summary
            keyword_query = supabase_client.from_('code_examples')\
                .select('id, url, chunk_number, content, summary, metadata, source_id')\
                .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')
            
            # Apply source filter if provided
            if source_id and source_id.strip():
                keyword_query = keyword_query.eq('source_id', source_id)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'summary': kr['summary'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            from utils import search_code_examples as search_code_examples_impl
            
            results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_id,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking and ctx.request_context.lifespan_context.reranking_model is not None,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10, adaptive: bool = False) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        adaptive: If True, stop crawling when sufficient content is gathered
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = []
    page_contents = []
    
    # If adaptive, process URLs in batches and check if we should continue
    if adaptive:
        for url in urls:
            # Safety check for each URL
            if not is_public_http_url(url):
                continue
                
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success and result.markdown:
                results.append({'url': result.url, 'markdown': result.markdown})
                page_contents.append(result.markdown)
                
                # Check if we should continue crawling
                if not should_continue_crawling(page_contents, len(urls)):
                    break
    else:
        # Non-adaptive: crawl all URLs in parallel
        # Filter URLs for safety first
        safe_urls = [u for u in urls if is_public_http_url(u)]
        if safe_urls:
            crawl_results = await crawler.arun_many(urls=safe_urls, config=crawl_config, dispatcher=dispatcher)
            results = [{'url': r.url, 'markdown': r.markdown} for r in crawl_results if r.success and r.markdown]
    
    return results

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10, adaptive: bool = False) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        adaptive: If True, stop crawling when sufficient content is gathered
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()
    page_contents = []

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls if is_public_http_url(u)])
    results_all = []

    for depth in range(max_depth):
        # Filter URLs for safety
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited and is_public_http_url(url)]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                page_contents.append(result.markdown)
                
                # Check adaptive stopping
                if adaptive and not should_continue_crawling(page_contents, len(urls_to_crawl) * max_depth):
                    return results_all
                
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    # Safety check for internal links
                    if next_url not in visited and is_public_http_url(next_url):
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

# =============================================================================
# SEO ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
async def get_raw_html(ctx: Context, url: str, max_length: int = 100000) -> str:
    """
    Get raw HTML content from a webpage (similar to Firecrawl's html format).
    
    This tool returns the raw HTML which is useful for:
    - Custom HTML analysis
    - Extracting specific elements
    - SEO auditing that requires HTML structure
    - Passing to other analysis pipelines
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to fetch
        max_length: Maximum HTML length to return (default: 100000 chars)
    
    Returns:
        JSON string with raw HTML and basic metadata
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to crawl URL')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        html = result.html or ""
        
        # Extract basic metadata from HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        title = soup.title.string if soup.title else None
        
        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else None
        
        return json.dumps({
            "success": True,
            "url": url,
            "html": html[:max_length],
            "html_length": len(html),
            "truncated": len(html) > max_length,
            "metadata": {
                "title": title,
                "description": description
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def extract_seo_metadata(ctx: Context, url: str) -> str:
    """
    Extract SEO metadata from a webpage including title, meta tags, Open Graph, and JSON-LD structured data.
    
    This tool is essential for SEO audits as it extracts:
    - Page title
    - Meta description, keywords, author, robots directives
    - Open Graph tags (og:title, og:description, og:image, etc.)
    - Twitter Card tags
    - JSON-LD structured data (Schema.org)
    - Canonical URL
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to analyze
    
    Returns:
        JSON string with all SEO metadata extracted from the page
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        # Safety check
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        # Get the crawler from context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # Crawl with full HTML access
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to crawl URL')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        # Parse the HTML to extract SEO elements
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else None
        
        # Extract meta tags
        meta_tags = {}
        og_tags = {}
        twitter_tags = {}
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            # Standard meta tags
            if name in ['description', 'keywords', 'author', 'robots', 'viewport', 'generator']:
                meta_tags[name] = content
            
            # Open Graph tags
            if property_attr.startswith('og:'):
                og_tags[property_attr] = content
            
            # Twitter Card tags
            if name.startswith('twitter:') or property_attr.startswith('twitter:'):
                key = name if name.startswith('twitter:') else property_attr
                twitter_tags[key] = content
        
        # Extract canonical URL
        canonical = None
        canonical_tag = soup.find('link', rel='canonical')
        if canonical_tag:
            canonical = canonical_tag.get('href')
        
        # Extract JSON-LD structured data
        jsonld_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                jsonld_data.append(data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Extract hreflang tags
        hreflang_tags = []
        for link in soup.find_all('link', rel='alternate'):
            hreflang = link.get('hreflang')
            if hreflang:
                hreflang_tags.append({
                    "hreflang": hreflang,
                    "href": link.get('href')
                })
        
        # SEO issues detection
        issues = []
        if not title:
            issues.append("Missing page title")
        elif len(title) > 60:
            issues.append(f"Title too long ({len(title)} chars, recommended: 50-60)")
        elif len(title) < 30:
            issues.append(f"Title too short ({len(title)} chars, recommended: 50-60)")
        
        description = meta_tags.get('description', '')
        if not description:
            issues.append("Missing meta description")
        elif len(description) > 160:
            issues.append(f"Meta description too long ({len(description)} chars, recommended: 150-160)")
        elif len(description) < 120:
            issues.append(f"Meta description too short ({len(description)} chars, recommended: 150-160)")
        
        if not canonical:
            issues.append("Missing canonical URL")
        
        if not og_tags:
            issues.append("Missing Open Graph tags")
        
        if not jsonld_data:
            issues.append("Missing structured data (JSON-LD)")
        
        return json.dumps({
            "success": True,
            "url": url,
            "title": title,
            "title_length": len(title) if title else 0,
            "meta_tags": meta_tags,
            "open_graph": og_tags,
            "twitter_cards": twitter_tags,
            "canonical_url": canonical,
            "structured_data": jsonld_data,
            "hreflang_tags": hreflang_tags,
            "seo_issues": issues,
            "issues_count": len(issues)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def extract_page_structure(ctx: Context, url: str) -> str:
    """
    Extract page structure for SEO analysis: headings hierarchy, images with alt text, and link analysis.
    
    This tool analyzes:
    - Heading structure (H1-H6) and their hierarchy
    - Images with/without alt text
    - Internal vs external links
    - Anchor text distribution
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to analyze
    
    Returns:
        JSON string with page structure analysis
    """
    try:
        # Validate URL
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to crawl URL')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result.html, 'html.parser')
        
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        
        # Extract headings
        headings = {"h1": [], "h2": [], "h3": [], "h4": [], "h5": [], "h6": []}
        for level in headings.keys():
            for h in soup.find_all(level):
                text = h.get_text(strip=True)
                if text:
                    headings[level].append(text)
        
        # Extract images
        images_with_alt = []
        images_without_alt = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            img_data = {"src": src, "alt": alt}
            if alt and alt.strip():
                images_with_alt.append(img_data)
            else:
                images_without_alt.append(img_data)
        
        # Extract links
        internal_links = []
        external_links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            rel = a.get('rel', [])
            
            link_data = {
                "href": href,
                "text": text[:100] if text else "",
                "nofollow": 'nofollow' in rel
            }
            
            # Determine if internal or external
            if href.startswith(('http://', 'https://')):
                link_domain = urlparse(href).netloc
                if link_domain == base_domain:
                    internal_links.append(link_data)
                else:
                    external_links.append(link_data)
            elif href.startswith('/') or not href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                internal_links.append(link_data)
        
        # SEO issues
        issues = []
        h1_count = len(headings['h1'])
        if h1_count == 0:
            issues.append("Missing H1 heading")
        elif h1_count > 1:
            issues.append(f"Multiple H1 headings ({h1_count}) - should have exactly one")
        
        if images_without_alt:
            issues.append(f"{len(images_without_alt)} images missing alt text")
        
        total_images = len(images_with_alt) + len(images_without_alt)
        
        return json.dumps({
            "success": True,
            "url": url,
            "headings": {
                "h1": headings["h1"],
                "h2": headings["h2"],
                "h3": headings["h3"][:10],  # Limit to avoid bloat
                "h4_count": len(headings["h4"]),
                "h5_count": len(headings["h5"]),
                "h6_count": len(headings["h6"])
            },
            "heading_counts": {level: len(items) for level, items in headings.items()},
            "images": {
                "total": total_images,
                "with_alt": len(images_with_alt),
                "without_alt": len(images_without_alt),
                "missing_alt_examples": images_without_alt[:5]
            },
            "links": {
                "internal_count": len(internal_links),
                "external_count": len(external_links),
                "internal_sample": internal_links[:10],
                "external_sample": external_links[:10]
            },
            "seo_issues": issues
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def check_broken_links(ctx: Context, url: str, max_links: int = 50) -> str:
    """
    Check for broken links on a webpage.
    
    This tool crawls a page, extracts all links, and checks their HTTP status codes
    to identify broken links (4xx, 5xx errors).
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to analyze
        max_links: Maximum number of links to check (default: 50)
    
    Returns:
        JSON string with broken link analysis
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to crawl URL')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        # Collect all links
        all_links = set()
        for link_type in ['internal', 'external']:
            for link in result.links.get(link_type, []):
                href = link.get('href', '') if isinstance(link, dict) else link
                if href and href.startswith(('http://', 'https://')):
                    # Safety check for each link
                    if is_public_http_url(href):
                        all_links.add(href)
        
        # Limit links to check
        links_to_check = list(all_links)[:max_links]
        
        # Check links in parallel
        import aiohttp
        broken_links = []
        redirects = []
        working_links = []
        
        async def check_single_link(session, link):
            try:
                async with session.head(link, allow_redirects=False) as response:
                    return {"url": link, "status": response.status, "location": response.headers.get('Location', '')}
            except Exception as e:
                return {"url": link, "status": "error", "error": str(e)[:50]}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = [check_single_link(session, link) for link in links_to_check]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    continue
                status = result.get("status")
                if status == "error":
                    broken_links.append(result)
                elif isinstance(status, int):
                    if status >= 400:
                        broken_links.append({"url": result["url"], "status": status})
                    elif status >= 300:
                        redirects.append({"url": result["url"], "status": status, "redirect_to": result.get("location", "")})
                    else:
                        working_links.append({"url": result["url"], "status": status})
        
        return json.dumps({
            "success": True,
            "url": url,
            "links_checked": len(links_to_check),
            "broken_links": broken_links,
            "broken_count": len(broken_links),
            "redirects": redirects,
            "redirect_count": len(redirects),
            "working_count": len(working_links)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def analyze_robots_txt(ctx: Context, url: str) -> str:
    """
    Analyze the robots.txt file for a domain.
    
    This tool fetches and parses the robots.txt file to show:
    - User-agent rules
    - Disallowed paths
    - Allowed paths
    - Sitemap locations
    - Crawl-delay directives
    
    Args:
        ctx: The MCP server provided context
        url: Any URL from the domain (robots.txt will be fetched from root)
    
    Returns:
        JSON string with robots.txt analysis
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        # Construct robots.txt URL
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Fetch robots.txt
        robots_content = await fetch_text(robots_url)
        
        if not robots_content:
            return json.dumps({
                "success": True,
                "url": robots_url,
                "exists": False,
                "message": "No robots.txt found"
            }, indent=2)
        
        # Parse robots.txt
        rules = {}
        sitemaps = []
        current_agent = None
        
        for line in robots_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                directive, value = line.split(':', 1)
                directive = directive.strip().lower()
                value = value.strip()
                
                if directive == 'user-agent':
                    current_agent = value
                    if current_agent not in rules:
                        rules[current_agent] = {"disallow": [], "allow": [], "crawl_delay": None}
                elif directive == 'disallow' and current_agent:
                    if value:
                        rules[current_agent]["disallow"].append(value)
                elif directive == 'allow' and current_agent:
                    if value:
                        rules[current_agent]["allow"].append(value)
                elif directive == 'crawl-delay' and current_agent:
                    rules[current_agent]["crawl_delay"] = value
                elif directive == 'sitemap':
                    sitemaps.append(value)
        
        # SEO issues
        issues = []
        if '*' not in rules:
            issues.append("No rules for generic user-agent (*)")
        
        if not sitemaps:
            issues.append("No sitemap declared in robots.txt")
        
        # Check for common bot restrictions
        ai_bots_blocked = []
        ai_bots = ['GPTBot', 'ChatGPT-User', 'CCBot', 'anthropic-ai', 'Claude-Web', 'Google-Extended']
        for bot in ai_bots:
            if bot in rules:
                if rules[bot].get("disallow") and '/' in rules[bot]["disallow"]:
                    ai_bots_blocked.append(bot)
        
        return json.dumps({
            "success": True,
            "url": robots_url,
            "exists": True,
            "rules": rules,
            "sitemaps": sitemaps,
            "ai_bots_blocked": ai_bots_blocked,
            "seo_issues": issues,
            "raw_content": robots_content[:2000]  # First 2000 chars
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def check_llms_txt(ctx: Context, url: str) -> str:
    """
    Check for llms.txt file which defines AI usage permissions for a website.
    
    This tool checks for various llms.txt file variations:
    - llms.txt
    - LLMs.txt  
    - llms-full.txt
    
    Args:
        ctx: The MCP server provided context
        url: Any URL from the domain (llms.txt will be fetched from root)
    
    Returns:
        JSON string with llms.txt analysis
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check all llms.txt variations
        filenames = ['llms.txt', 'LLMs.txt', 'llms-full.txt']
        
        for filename in filenames:
            try:
                llms_url = f"{base_url}/{filename}"
                content = await fetch_text(llms_url)
                
                if content:
                    # Verify it's actually an LLMs.txt file, not a 404 page
                    is_valid = (
                        len(content) > 10 and
                        '<!DOCTYPE' not in content and
                        '<html' not in content.lower() and
                        '404' not in content.lower()[:100] and
                        'not found' not in content.lower()[:100]
                    )
                    
                    if is_valid:
                        return json.dumps({
                            "success": True,
                            "url": llms_url,
                            "exists": True,
                            "filename": filename,
                            "content_length": len(content),
                            "content_preview": content[:2000],
                            "seo_issues": []
                        }, indent=2)
            except:
                continue
        
        return json.dumps({
            "success": True,
            "url": base_url,
            "exists": False,
            "message": "No llms.txt file found",
            "seo_issues": ["No llms.txt file - consider adding AI usage guidelines"]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def analyze_accessibility(ctx: Context, url: str) -> str:
    """
    Analyze webpage accessibility features important for AI comprehension.
    
    This tool checks:
    - ARIA labels and roles
    - Lang attribute
    - Alt text coverage
    - Form labels
    - Skip links
    - Color contrast indicators
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to analyze
    
    Returns:
        JSON string with accessibility analysis
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to crawl URL')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Check lang attribute
        html_tag = soup.find('html')
        has_lang = html_tag.get('lang') if html_tag else None
        
        # Check ARIA
        aria_labels = soup.find_all(attrs={'aria-label': True})
        aria_describedby = soup.find_all(attrs={'aria-describedby': True})
        aria_roles = soup.find_all(attrs={'role': True})
        
        # Check images
        images = soup.find_all('img')
        images_with_alt = [img for img in images if img.get('alt', '').strip()]
        images_without_alt = [img for img in images if not img.get('alt', '').strip()]
        
        # Check form accessibility
        forms = soup.find_all('form')
        inputs = soup.find_all(['input', 'select', 'textarea'])
        inputs_with_label = []
        inputs_without_label = []
        
        for inp in inputs:
            inp_id = inp.get('id')
            inp_type = inp.get('type', '')
            if inp_type in ['hidden', 'submit', 'button']:
                continue
            has_label = (
                inp.get('aria-label') or
                inp.get('aria-labelledby') or
                (inp_id and soup.find('label', attrs={'for': inp_id}))
            )
            if has_label:
                inputs_with_label.append(inp_id or inp_type)
            else:
                inputs_without_label.append(inp_id or inp_type)
        
        # Check skip links
        skip_links = soup.find_all('a', href=lambda x: x and x.startswith('#'))
        has_skip_to_content = any('skip' in (link.get_text() or '').lower() or 'main' in (link.get('href') or '').lower() 
                                   for link in skip_links[:5])
        
        # Calculate score
        score = 0
        issues = []
        
        # Lang attribute (15 points)
        if has_lang:
            score += 15
        else:
            issues.append("Missing lang attribute on <html>")
        
        # Alt text (30 points)
        if images:
            alt_ratio = len(images_with_alt) / len(images)
            score += int(alt_ratio * 30)
            if alt_ratio < 1:
                issues.append(f"{len(images_without_alt)} images missing alt text")
        else:
            score += 30  # No images = no penalty
        
        # ARIA (25 points)
        aria_score = min(25, len(aria_labels) * 3 + len(aria_roles) * 2 + len(aria_describedby) * 2)
        score += aria_score
        if aria_score < 10:
            issues.append("Limited ARIA labels/roles for screen readers")
        
        # Form accessibility (15 points)
        if inputs:
            label_ratio = len(inputs_with_label) / max(1, len(inputs_with_label) + len(inputs_without_label))
            score += int(label_ratio * 15)
            if inputs_without_label:
                issues.append(f"{len(inputs_without_label)} form inputs missing labels")
        else:
            score += 15
        
        # Skip links (15 points)
        if has_skip_to_content:
            score += 15
        else:
            issues.append("No skip-to-content link found")
        
        score = min(100, score)
        
        return json.dumps({
            "success": True,
            "url": url,
            "accessibility_score": score,
            "lang_attribute": has_lang,
            "aria": {
                "labels_count": len(aria_labels),
                "roles_count": len(aria_roles),
                "describedby_count": len(aria_describedby)
            },
            "images": {
                "total": len(images),
                "with_alt": len(images_with_alt),
                "without_alt": len(images_without_alt),
                "missing_alt_examples": [img.get('src', '')[:50] for img in images_without_alt[:3]]
            },
            "forms": {
                "total_forms": len(forms),
                "inputs_with_labels": len(inputs_with_label),
                "inputs_without_labels": len(inputs_without_label)
            },
            "has_skip_link": has_skip_to_content,
            "seo_issues": issues
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def analyze_readability(ctx: Context, url: str) -> str:
    """
    Analyze content readability using Flesch-Kincaid score.
    
    This tool calculates readability metrics important for AI comprehension:
    - Flesch Reading Ease score (0-100, higher = easier)
    - Average words per sentence
    - Average syllables per word
    - Content complexity assessment
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to analyze
    
    Returns:
        JSON string with readability analysis
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to crawl URL')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        # Use markdown for cleaner text - handle both string and MarkdownGenerationResult
        markdown_result = result.markdown
        if hasattr(markdown_result, 'raw_markdown'):
            text = markdown_result.raw_markdown or ""
        else:
            text = str(markdown_result) if markdown_result else ""
        
        # Remove code blocks for readability analysis
        import re
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Count sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 3]
        
        # Count words
        words = [w for w in re.split(r'\s+', text) if w and len(w) > 0]
        
        # Count syllables (simple approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiou'
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            # Handle silent e
            if word.endswith('e') and count > 1:
                count -= 1
            return max(1, count)
        
        total_syllables = sum(count_syllables(w) for w in words)
        
        # Calculate metrics
        num_sentences = max(1, len(sentences))
        num_words = max(1, len(words))
        
        avg_words_per_sentence = num_words / num_sentences
        avg_syllables_per_word = total_syllables / num_words
        
        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))
        
        # Determine readability level
        if flesch_score >= 70:
            level = "Easy"
            status = "pass"
            description = "Content is easily readable, ideal for AI comprehension"
        elif flesch_score >= 50:
            level = "Moderate"
            status = "pass"
            description = "Content is fairly readable"
        elif flesch_score >= 30:
            level = "Difficult"
            status = "warning"
            description = "Content may be hard to process accurately"
        else:
            level = "Very Difficult"
            status = "fail"
            description = "Content is very complex, may reduce AI accuracy"
        
        issues = []
        if flesch_score < 50:
            issues.append(f"Low readability score ({flesch_score:.0f}) - consider simplifying content")
        if avg_words_per_sentence > 25:
            issues.append(f"Long sentences (avg {avg_words_per_sentence:.1f} words) - consider shorter sentences")
        
        return json.dumps({
            "success": True,
            "url": url,
            "flesch_reading_ease": round(flesch_score, 1),
            "readability_level": level,
            "status": status,
            "description": description,
            "metrics": {
                "total_words": num_words,
                "total_sentences": num_sentences,
                "total_syllables": total_syllables,
                "avg_words_per_sentence": round(avg_words_per_sentence, 1),
                "avg_syllables_per_word": round(avg_syllables_per_word, 2)
            },
            "seo_issues": issues
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def full_seo_audit(ctx: Context, url: str) -> str:
    """
    Perform a comprehensive SEO audit combining all SEO analysis tools.
    
    This tool runs all SEO checks in one call:
    - Meta tags and structured data
    - Page structure (headings, images, links)
    - Robots.txt analysis
    - llms.txt check
    - Accessibility analysis
    - Readability score
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to audit
    
    Returns:
        JSON string with complete SEO audit results
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        # Run all checks in parallel
        results = await asyncio.gather(
            extract_seo_metadata(ctx, url),
            extract_page_structure(ctx, url),
            analyze_robots_txt(ctx, url),
            check_llms_txt(ctx, url),
            analyze_accessibility(ctx, url),
            analyze_readability(ctx, url)
        )
        
        metadata_result = json.loads(results[0])
        structure_result = json.loads(results[1])
        robots_result = json.loads(results[2])
        llms_result = json.loads(results[3])
        accessibility_result = json.loads(results[4])
        readability_result = json.loads(results[5])
        
        # Combine all issues
        all_issues = []
        for result in [metadata_result, structure_result, robots_result, llms_result, accessibility_result, readability_result]:
            if result.get("success"):
                all_issues.extend(result.get("seo_issues", []))
        
        # Calculate weighted score
        weights = {
            "readability": 1.5,
            "heading_structure": 1.4,
            "meta_tags": 1.2,
            "robots": 0.9,
            "sitemap": 0.8,
            "llms": 0.3,
            "semantic_html": 1.0,
            "accessibility": 0.9
        }
        
        scores = {
            "readability": readability_result.get("flesch_reading_ease", 0) if readability_result.get("success") else 0,
            "accessibility": accessibility_result.get("accessibility_score", 0) if accessibility_result.get("success") else 0,
            "meta_tags": 100 - (metadata_result.get("issues_count", 5) * 15) if metadata_result.get("success") else 0,
            "structure": 100 - (len(structure_result.get("seo_issues", [])) * 20) if structure_result.get("success") else 0,
            "robots": 100 if robots_result.get("exists") else 0,
            "llms": 100 if llms_result.get("exists") else 0
        }
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        weighted_sum = (
            scores["readability"] * weights["readability"] +
            scores["accessibility"] * weights["accessibility"] +
            scores["meta_tags"] * weights["meta_tags"] +
            scores["structure"] * weights["heading_structure"] +
            scores["robots"] * weights["robots"] +
            scores["llms"] * weights["llms"]
        )
        
        overall_score = max(0, min(100, int(weighted_sum / total_weight)))
        
        return json.dumps({
            "success": True,
            "url": url,
            "seo_score": overall_score,
            "total_issues": len(all_issues),
            "all_issues": all_issues,
            "component_scores": scores,
            "metadata": metadata_result if metadata_result.get("success") else {"error": metadata_result.get("error")},
            "structure": structure_result if structure_result.get("success") else {"error": structure_result.get("error")},
            "robots": robots_result if robots_result.get("success") else {"error": robots_result.get("error")},
            "llms_txt": llms_result if llms_result.get("success") else {"error": llms_result.get("error")},
            "accessibility": accessibility_result if accessibility_result.get("success") else {"error": accessibility_result.get("error")},
            "readability": readability_result if readability_result.get("success") else {"error": readability_result.get("error")}
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# =============================================================================
# BROWSER AUTOMATION TOOLS
# =============================================================================

@mcp.tool()
async def capture_screenshot(ctx: Context, url: str, output_path: str = None, full_page: bool = True) -> str:
    """
    Capture a screenshot of a webpage as PNG.
    
    This tool renders the page in a headless browser and captures a screenshot.
    If output_path is provided, saves to disk; otherwise returns base64 data.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to screenshot
        output_path: Optional file path to save PNG (e.g., "./screenshot.png")
        full_page: If True, capture the full scrollable page; if False, capture only the viewport
    
    Returns:
        JSON with file path if saved, or base64 data if no output_path
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        
        run_config = CrawlerRunConfig(
            screenshot=True,
            cache_mode=CacheMode.BYPASS,
            stream=False
        )
        
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to capture screenshot')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        screenshot_data = result.screenshot if hasattr(result, 'screenshot') else None
        
        if not screenshot_data:
            return json.dumps({
                "success": False, 
                "error": "Screenshot capture failed - no data returned"
            }, indent=2)
        
        response = {
            "success": True,
            "url": url,
            "screenshot": screenshot_data,
            "format": "base64_png",
            "full_page": full_page
        }
        
        # If output_path provided, also save to file
        if output_path:
            import base64
            img_bytes = base64.b64decode(screenshot_data)
            
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(img_bytes)
            
            response["saved_to"] = os.path.abspath(output_path)
            response["file_size_bytes"] = len(img_bytes)
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def generate_pdf(ctx: Context, url: str, output_path: str = None) -> str:
    """
    Generate a PDF of a webpage.
    
    This tool renders the page in a headless browser and generates a PDF document.
    If output_path is provided, saves to disk; otherwise returns base64 data.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to convert to PDF
        output_path: Optional file path to save PDF (e.g., "./page.pdf")
    
    Returns:
        JSON with file path if saved, or base64 data if no output_path
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        
        run_config = CrawlerRunConfig(
            pdf=True,
            cache_mode=CacheMode.BYPASS,
            stream=False
        )
        
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to generate PDF')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        pdf_data = result.pdf if hasattr(result, 'pdf') else None
        
        if not pdf_data:
            return json.dumps({
                "success": False, 
                "error": "PDF generation failed - no data returned"
            }, indent=2)
        
        # Convert to base64 if bytes
        import base64
        if isinstance(pdf_data, bytes):
            pdf_bytes = pdf_data
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        else:
            pdf_base64 = pdf_data
            pdf_bytes = base64.b64decode(pdf_data)
        
        response = {
            "success": True,
            "url": url,
            "pdf": pdf_base64,
            "format": "base64_pdf"
        }
        
        # If output_path provided, also save to file
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            
            response["saved_to"] = os.path.abspath(output_path)
            response["file_size_bytes"] = len(pdf_bytes)
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def execute_javascript(ctx: Context, url: str, scripts: List[str]) -> str:
    """
    Execute custom JavaScript on a webpage and return the results.
    
    This tool loads a page and executes one or more JavaScript snippets in sequence.
    Useful for:
    - Extracting data from JavaScript-rendered pages
    - Interacting with dynamic content
    - Testing page behavior
    - Scraping SPAs (Single Page Applications)
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to execute JavaScript on
        scripts: List of JavaScript code snippets to execute sequentially
    
    Returns:
        JSON string with execution results and page content
    
    Example scripts:
        - "return document.title"
        - "return Array.from(document.querySelectorAll('a')).map(a => a.href)"
        - "document.querySelector('button.load-more').click()"
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not scripts or len(scripts) == 0:
            return json.dumps({"success": False, "error": "At least one script is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        
        run_config = CrawlerRunConfig(
            js_code=scripts,
            cache_mode=CacheMode.BYPASS,
            stream=False
        )
        
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to execute JavaScript')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        # Get markdown content
        markdown_content = ""
        if hasattr(result, 'markdown'):
            if hasattr(result.markdown, 'raw_markdown'):
                markdown_content = result.markdown.raw_markdown
            else:
                markdown_content = str(result.markdown) if result.markdown else ""
        
        return json.dumps({
            "success": True,
            "url": url,
            "scripts_executed": len(scripts),
            "markdown": markdown_content[:50000] if markdown_content else "",
            "html_length": len(result.html) if result.html else 0,
            "message": f"Successfully executed {len(scripts)} script(s) on the page."
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def convert_actions_to_js(actions: List[Dict[str, Any]]) -> List[str]:
    """
    Convert a list of action dictionaries to JavaScript code snippets.
    
    Supported action types:
    - click: Click an element by selector
    - type: Type text into an input field
    - scroll: Scroll the page
    - wait: Wait for a specified time
    - wait_for: Wait for an element to appear
    
    Args:
        actions: List of action dictionaries
        
    Returns:
        List of JavaScript code strings
    """
    js_scripts = []
    
    for action in actions:
        action_type = action.get("type", "").lower()
        
        if action_type == "click":
            selector = action.get("selector", "")
            if selector:
                js_scripts.append(f"document.querySelector('{selector}')?.click();")
        
        elif action_type == "type":
            selector = action.get("selector", "")
            text = action.get("text", "")
            if selector and text:
                # Escape single quotes in text
                escaped_text = text.replace("'", "\\'")
                js_scripts.append(f"""
                    const el = document.querySelector('{selector}');
                    if (el) {{
                        el.value = '{escaped_text}';
                        el.dispatchEvent(new Event('input', {{bubbles: true}}));
                    }}
                """)
        
        elif action_type == "scroll":
            direction = action.get("direction", "down").lower()
            amount = action.get("amount", 500)
            if direction == "down":
                js_scripts.append(f"window.scrollBy(0, {amount});")
            elif direction == "up":
                js_scripts.append(f"window.scrollBy(0, -{amount});")
            elif direction == "bottom":
                js_scripts.append("window.scrollTo(0, document.body.scrollHeight);")
            elif direction == "top":
                js_scripts.append("window.scrollTo(0, 0);")
        
        elif action_type == "wait":
            milliseconds = action.get("milliseconds", 1000)
            js_scripts.append(f"await new Promise(r => setTimeout(r, {milliseconds}));")
        
        elif action_type == "wait_for":
            selector = action.get("selector", "")
            timeout = action.get("timeout", 5000)
            if selector:
                js_scripts.append(f"""
                    await new Promise((resolve, reject) => {{
                        const startTime = Date.now();
                        const checkInterval = setInterval(() => {{
                            if (document.querySelector('{selector}')) {{
                                clearInterval(checkInterval);
                                resolve();
                            }} else if (Date.now() - startTime > {timeout}) {{
                                clearInterval(checkInterval);
                                resolve();
                            }}
                        }}, 100);
                    }});
                """)
        
        elif action_type == "press":
            key = action.get("key", "Enter")
            selector = action.get("selector", "")
            if selector:
                js_scripts.append(f"""
                    const pressEl = document.querySelector('{selector}');
                    if (pressEl) {{
                        pressEl.dispatchEvent(new KeyboardEvent('keydown', {{
                            key: '{key}', keyCode: {{'Enter': 13, 'Tab': 9, 'Escape': 27}}.get('{key}', 13), bubbles: true
                        }}));
                    }}
                """)
            else:
                js_scripts.append(f"""
                    document.dispatchEvent(new KeyboardEvent('keydown', {{
                        key: '{key}', bubbles: true
                    }}));
                """)
    
    return js_scripts


@mcp.tool()
async def crawl_with_actions(ctx: Context, url: str, actions: List[Dict[str, Any]], screenshot: bool = False) -> str:
    """
    Crawl a webpage after performing browser actions (click, type, scroll, wait).
    
    This tool is essential for scraping dynamic content that requires user interaction,
    such as clicking "Load More" buttons, filling forms, or navigating SPAs.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the webpage to interact with
        actions: List of actions to perform before scraping. Each action is a dict with:
            - type: "click" | "type" | "scroll" | "wait" | "wait_for" | "press"
            - selector: CSS selector for the target element (for click, type, wait_for, press)
            - text: Text to type (for type action)
            - direction: "up" | "down" | "top" | "bottom" (for scroll)
            - amount: Pixels to scroll (for scroll, default 500)
            - milliseconds: Time to wait (for wait action)
            - timeout: Max time to wait for element (for wait_for, default 5000)
            - key: Key to press (for press action, e.g., "Enter", "Tab")
        screenshot: If True, also capture a screenshot after actions
    
    Returns:
        JSON string with page content after actions, optionally with screenshot
    
    Example actions:
        [
            {"type": "click", "selector": "button.accept-cookies"},
            {"type": "wait", "milliseconds": 500},
            {"type": "type", "selector": "input#search", "text": "AI tools"},
            {"type": "press", "selector": "input#search", "key": "Enter"},
            {"type": "wait_for", "selector": ".search-results"},
            {"type": "scroll", "direction": "down", "amount": 1000}
        ]
    """
    try:
        if not url or not url.strip():
            return json.dumps({"success": False, "error": "URL is required"}, indent=2)
        
        if not actions or len(actions) == 0:
            return json.dumps({"success": False, "error": "At least one action is required"}, indent=2)
        
        if not url.startswith(('http://', 'https://')):
            return json.dumps({"success": False, "error": "Invalid URL format"}, indent=2)
        
        try:
            require_public_http_url(url)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)
        
        # Convert actions to JavaScript
        js_scripts = convert_actions_to_js(actions)
        
        if not js_scripts:
            return json.dumps({"success": False, "error": "No valid actions provided"}, indent=2)
        
        crawler = ctx.request_context.lifespan_context.crawler
        
        run_config = CrawlerRunConfig(
            js_code=js_scripts,
            screenshot=screenshot,
            cache_mode=CacheMode.BYPASS,
            stream=False
        )
        
        result = await crawler.arun(url=url, config=run_config)
        
        if not result.success:
            error_msg = getattr(result, 'error_message', 'Failed to execute actions')
            return json.dumps({"success": False, "error": error_msg}, indent=2)
        
        # Get markdown content
        markdown_content = ""
        if hasattr(result, 'markdown'):
            if hasattr(result.markdown, 'raw_markdown'):
                markdown_content = result.markdown.raw_markdown
            else:
                markdown_content = str(result.markdown) if result.markdown else ""
        
        response = {
            "success": True,
            "url": url,
            "actions_performed": len(actions),
            "markdown": markdown_content[:50000] if markdown_content else "",
            "html_length": len(result.html) if result.html else 0
        }
        
        if screenshot and hasattr(result, 'screenshot') and result.screenshot:
            response["screenshot"] = result.screenshot
            response["screenshot_format"] = "base64_png"
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())