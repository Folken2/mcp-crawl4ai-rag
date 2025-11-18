#!/usr/bin/env python3
"""
Test MCP server tools using the MCP client library via SSE.
This properly tests the server as it would be used by an MCP client.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    print("MCP client library not found. Using direct test instead.")
    print("Run: uv pip install mcp")
    sys.exit(1)

async def test_tools_via_sse():
    """Test tools via SSE connection"""
    print("=" * 60)
    print("Testing MCP Server Tools via SSE")
    print("=" * 60)
    
    server_url = "http://localhost:8051/sse"
    
    try:
        # Connect to the SSE server
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # List available tools
                print("\n1. Listing available tools...")
                tools_result = await session.list_tools()
                print(f"   Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    print(f"     - {tool.name}: {tool.description[:60]}...")
                
                # Test crawl_single_page_raw
                print("\n2. Testing crawl_single_page_raw...")
                try:
                    result = await session.call_tool(
                        "crawl_single_page_raw",
                        arguments={"url": "https://celofixings.com"}
                    )
                    
                    if result.content:
                        # Parse the JSON response
                        content_text = result.content[0].text if result.content else ""
                        result_data = json.loads(content_text)
                        
                        if result_data.get("success"):
                            print(f"   ✓ Successfully crawled {result_data.get('url')}")
                            print(f"     Content length: {result_data.get('content_length')} characters")
                            print(f"     Word count: {result_data.get('word_count')}")
                            print(f"     Internal links: {result_data.get('links_count', {}).get('internal', 0)}")
                            print(f"     External links: {result_data.get('links_count', {}).get('external', 0)}")
                            print(f"\n     Markdown preview (first 300 chars):")
                            print(f"     {result_data.get('markdown', '')[:300]}...")
                            
                            # Save markdown to file
                            url = result_data.get('url', 'unknown').replace('https://', '').replace('http://', '').replace('/', '_')
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"crawl_single_{url}_{timestamp}.md"
                            output_path = Path(__file__).parent / filename
                            
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(f"# Crawled: {result_data.get('url')}\n\n")
                                f.write(f"**Crawled at:** {datetime.now().isoformat()}\n\n")
                                f.write(f"**Content length:** {result_data.get('content_length')} characters\n")
                                f.write(f"**Word count:** {result_data.get('word_count')} words\n")
                                f.write(f"**Internal links:** {result_data.get('links_count', {}).get('internal', 0)}\n")
                                f.write(f"**External links:** {result_data.get('links_count', {}).get('external', 0)}\n\n")
                                f.write("---\n\n")
                                f.write(result_data.get('markdown', ''))
                            
                            print(f"\n     ✓ Saved markdown to: {filename}")
                        else:
                            print(f"   ✗ Failed: {result_data.get('error')}")
                except Exception as e:
                    print(f"   ✗ Error: {e}")
                
                # Test smart_crawl_url_raw
                print("\n3. Testing smart_crawl_url_raw...")
                try:
                    result = await session.call_tool(
                        "smart_crawl_url_raw",
                        arguments={
                            "url": "https://celofixings.com",
                            "max_depth": 1,
                            "max_concurrent": 3
                        }
                    )
                    
                    if result.content:
                        # Parse the JSON response
                        content_text = result.content[0].text if result.content else ""
                        result_data = json.loads(content_text)
                        
                        if result_data.get("success"):
                            print(f"   ✓ Successfully crawled {result_data.get('url')}")
                            print(f"     Crawl type: {result_data.get('crawl_type')}")
                            print(f"     Pages crawled: {result_data.get('pages_crawled')}")
                            print(f"     Total content length: {result_data.get('total_content_length')} characters")
                            
                            pages = result_data.get('pages', [])
                            if pages:
                                print(f"\n     All URLs found ({len(pages)} pages):")
                                for i, page in enumerate(pages, 1):
                                    print(f"       {i}. {page.get('url')} ({page.get('content_length')} chars, {page.get('word_count')} words)")
                                
                                print(f"\n     First page preview:")
                                first_page = pages[0]
                                print(f"       URL: {first_page.get('url')}")
                                print(f"       Content length: {first_page.get('content_length')} characters")
                                print(f"       Markdown preview (first 200 chars):")
                                print(f"       {first_page.get('markdown', '')[:200]}...")
                                
                                # Save all pages to a markdown file
                                url = result_data.get('url', 'unknown').replace('https://', '').replace('http://', '').replace('/', '_')
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"smart_crawl_{url}_{timestamp}.md"
                                output_path = Path(__file__).parent / filename
                                
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(f"# Smart Crawl: {result_data.get('url')}\n\n")
                                    f.write(f"**Crawled at:** {datetime.now().isoformat()}\n\n")
                                    f.write(f"**Crawl type:** {result_data.get('crawl_type')}\n")
                                    f.write(f"**Pages crawled:** {result_data.get('pages_crawled')}\n")
                                    f.write(f"**Total content length:** {result_data.get('total_content_length')} characters\n\n")
                                    f.write("## URLs Found\n\n")
                                    for i, page in enumerate(pages, 1):
                                        f.write(f"{i}. [{page.get('url')}]({page.get('url')}) - {page.get('content_length')} chars, {page.get('word_count')} words\n")
                                    f.write("\n---\n\n")
                                    
                                    # Write each page's content
                                    for i, page in enumerate(pages, 1):
                                        f.write(f"## Page {i}: {page.get('url')}\n\n")
                                        f.write(f"**Content length:** {page.get('content_length')} characters  \n")
                                        f.write(f"**Word count:** {page.get('word_count')} words\n\n")
                                        f.write("---\n\n")
                                        f.write(page.get('markdown', ''))
                                        f.write("\n\n---\n\n")
                                
                                print(f"\n     ✓ Saved all pages to: {filename}")
                        else:
                            print(f"   ✗ Failed: {result_data.get('error')}")
                except Exception as e:
                    print(f"   ✗ Error: {e}")
                    
    except Exception as e:
        print(f"\n✗ Connection error: {e}")
        print("\nNote: Make sure the server is running on http://localhost:8051")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    print("\n" + "=" * 60)
    print("MCP Server Tools Test (via SSE)")
    print("=" * 60)
    print("Server: http://localhost:8051/sse\n")
    
    success = await test_tools_via_sse()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test completed")
    else:
        print("✗ Test failed")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

