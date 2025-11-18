#!/usr/bin/env python3
"""
Direct test of crawling tools - bypasses HTTP/SSE and tests tools directly.
This is simpler for testing the crawling functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai_mcp import crawl_single_page_raw, smart_crawl_url_raw, Crawl4AIContext
from mcp.server.fastmcp import Context
from unittest.mock import Mock

async def test_crawl_single_page_raw():
    """Test crawl_single_page_raw tool directly"""
    print("=" * 60)
    print("Testing crawl_single_page_raw")
    print("=" * 60)
    
    # Create a mock context
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    try:
        context = Crawl4AIContext(
            crawler=crawler,
            supabase_client=None,
            reranking_model=None
        )
        
        # Create a mock MCP context
        mock_request_context = Mock()
        mock_request_context.lifespan_context = context
        mock_ctx = Mock()
        mock_ctx.request_context = mock_request_context
        
        # Test the tool
        result = await crawl_single_page_raw(mock_ctx, "https://example.com")
        
        print("\nResult:")
        print(result)
        
        # Parse and display
        import json
        try:
            result_data = json.loads(result)
            if result_data.get("success"):
                print(f"\n✓ Successfully crawled {result_data.get('url')}")
                print(f"  Content length: {result_data.get('content_length')} characters")
                print(f"  Word count: {result_data.get('word_count')}")
                print(f"  Internal links: {result_data.get('links_count', {}).get('internal', 0)}")
                print(f"  External links: {result_data.get('links_count', {}).get('external', 0)}")
                print(f"\n  Markdown preview (first 300 chars):")
                print(f"  {result_data.get('markdown', '')[:300]}...")
                return True
            else:
                print(f"\n✗ Failed: {result_data.get('error')}")
                return False
        except json.JSONDecodeError:
            print("Could not parse result as JSON")
            return False
            
    finally:
        await crawler.__aexit__(None, None, None)

async def test_smart_crawl_url_raw():
    """Test smart_crawl_url_raw tool directly"""
    print("\n" + "=" * 60)
    print("Testing smart_crawl_url_raw")
    print("=" * 60)
    
    # Create a mock context
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    try:
        context = Crawl4AIContext(
            crawler=crawler,
            supabase_client=None,
            reranking_model=None
        )
        
        # Create a mock MCP context
        mock_request_context = Mock()
        mock_request_context.lifespan_context = context
        mock_ctx = Mock()
        mock_ctx.request_context = mock_request_context
        
        # Test the tool with shallow depth
        result = await smart_crawl_url_raw(mock_ctx, "https://example.com", max_depth=1, max_concurrent=3)
        
        print("\nResult:")
        print(result)
        
        # Parse and display
        import json
        try:
            result_data = json.loads(result)
            if result_data.get("success"):
                print(f"\n✓ Successfully crawled {result_data.get('url')}")
                print(f"  Crawl type: {result_data.get('crawl_type')}")
                print(f"  Pages crawled: {result_data.get('pages_crawled')}")
                print(f"  Total content length: {result_data.get('total_content_length')} characters")
                
                pages = result_data.get('pages', [])
                if pages:
                    print(f"\n  First page preview:")
                    first_page = pages[0]
                    print(f"    URL: {first_page.get('url')}")
                    print(f"    Content length: {first_page.get('content_length')} characters")
                    print(f"    Markdown preview (first 200 chars):")
                    print(f"    {first_page.get('markdown', '')[:200]}...")
                return True
            else:
                print(f"\n✗ Failed: {result_data.get('error')}")
                return False
        except json.JSONDecodeError:
            print("Could not parse result as JSON")
            return False
            
    finally:
        await crawler.__aexit__(None, None, None)

async def main():
    print("\n" + "=" * 60)
    print("Direct Crawling Tools Test")
    print("=" * 60)
    print("Testing tools directly (bypassing HTTP/SSE)\n")
    
    results = []
    
    # Test single page crawl
    results.append(await test_crawl_single_page_raw())
    
    # Test smart crawl
    results.append(await test_smart_crawl_url_raw())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

