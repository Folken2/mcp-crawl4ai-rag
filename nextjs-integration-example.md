# Next.js Integration Guide

Yes, you can use this MCP server from a Next.js app! The server uses the MCP (Model Context Protocol) over SSE (Server-Sent Events), which can be accessed from any HTTP client.

## Quick Answer

**Yes, you can use it from Next.js**, but you have two options:

1. **Use the MCP client library** (proper way, requires understanding MCP protocol)
2. **Create a Next.js API route that uses Python** (easier, since the MCP SDK is Python-based)

## Option 1: Python API Route in Next.js (Recommended)

Since the MCP SDK is Python-based, create a Python API route in Next.js:

### Install Python dependencies in your Next.js project

```bash
# In your Next.js project root
pip install mcp requests
```

### Create a Python API wrapper

Create `api/crawl.py`:

```python
import asyncio
import json
import sys
from mcp import ClientSession
from mcp.client.sse import sse_client

async def crawl_single_page(url: str):
    """Crawl a single page using MCP server"""
    server_url = "http://localhost:8051/sse"
    
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "crawl_single_page_raw",
                arguments={"url": url}
            )
            
            if result.content:
                content_text = result.content[0].text
                return json.loads(content_text)
            return {"success": False, "error": "No content"}

# For Next.js to call
if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    result = asyncio.run(crawl_single_page(url))
    print(json.dumps(result))
```

### Create Next.js API Route

Create `app/api/crawl/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const { url } = await request.json();
    
    if (!url) {
      return NextResponse.json(
        { success: false, error: 'URL is required' },
        { status: 400 }
      );
    }

    // Call Python script
    const { stdout } = await execAsync(
      `python3 api/crawl.py "${url}"`
    );
    
    const result = JSON.parse(stdout);
    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
```

## Option 2: Direct HTTP with MCP Protocol (Advanced)

If you want to implement the MCP protocol directly in TypeScript:

```typescript
// app/api/crawl/route.ts
import { NextRequest, NextResponse } from 'next/server';

const SERVER_URL = process.env.CRAWLER_SERVER_URL || 'http://localhost:8051/sse';

export async function POST(request: NextRequest) {
  try {
    const { url, tool = 'crawl_single_page_raw' } = await request.json();

    // MCP protocol uses SSE for bidirectional communication
    // This is complex - better to use Option 1 or a Python wrapper
    
    // For now, you'd need to:
    // 1. Establish SSE connection
    // 2. Send JSON-RPC messages
    // 3. Parse SSE responses
    
    return NextResponse.json({
      success: false,
      error: 'Use Python wrapper or MCP client library',
      note: 'See nextjs-integration-example.md for full implementation'
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
```

## Option 3: Use Existing Test Script as Reference

Look at `test_mcp_client.py` in this repository - it shows exactly how to connect and call tools. You can adapt that pattern for Next.js.

## Simplest Approach: Next.js API Route → Python Script → MCP Server

1. Next.js frontend calls `/api/crawl`
2. Next.js API route executes Python script
3. Python script uses MCP client to call server
4. Response flows back through the chain

## Environment Variables

Add to `.env.local`:

```
CRAWLER_SERVER_URL=http://localhost:8051/sse
```

## Available Tools

You can call any of these tools:
- `crawl_single_page_raw` - Single page, no Supabase needed
- `smart_crawl_url_raw` - Multi-page crawl, no Supabase needed
- `crawl_single_page` - Single page with Supabase storage
- `smart_crawl_url` - Multi-page with Supabase storage
- `get_available_sources` - List crawled sources
- `perform_rag_query` - Search stored content
- `search_code_examples` - Search code examples

## Example Frontend Component

```typescript
// app/components/Crawler.tsx
'use client';

import { useState } from 'react';

export default function Crawler() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleCrawl = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/crawl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="Enter URL to crawl"
      />
      <button onClick={handleCrawl} disabled={loading}>
        {loading ? 'Crawling...' : 'Crawl'}
      </button>
      {result && (
        <div>
          <h3>Result:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
```

## Important Notes

- The MCP server must be running (on localhost:8051 or your server URL)
- For production, deploy the MCP server separately and update the URL
- The `_raw` tools don't require Supabase, making them perfect for Next.js apps
- All tools return JSON, making them easy to work with in TypeScript/JavaScript
