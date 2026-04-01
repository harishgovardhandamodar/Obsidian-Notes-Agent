# mcp_server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from rag_engine import query_rag, query_by_page, get_pdf_info
import os

server = Server("local-codex-rag")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_knowledge_base",
            description="Search the local document vector store for context.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"]
            }
        ),
        Tool(
            name="query_pdf_page",
            description="Query a specific page number from indexed PDFs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "Page number"},
                    "context": {"type": "string", "description": "Optional context"}
                },
                "required": ["page"]
            }
        ),
        Tool(
            name="get_pdf_info",
            description="Get information about indexed PDFs.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="read_file",
            description="Read a file from the local filesystem.",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"]
            }
        ),
    ]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "search_knowledge_base":
        return [TextContent(type="text", text=query_rag(arguments["query"]))]
    
    elif name == "query_pdf_page":
        page = arguments.get("page")
        context = arguments.get("context", "")
        return [TextContent(type="text", text=query_by_page(page, context))]
    
    elif name == "get_pdf_info":
        return [TextContent(type="text", text=str(get_pdf_info()))]
    
    elif name == "read_file":
        path = arguments.get("path")
        if os.path.exists(path):
            with open(path, "r") as f:
                return [TextContent(type="text", text=f.read())]
        return [TextContent(type="text", text="File not found")]
    
    return []

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, None)

if __name__ == "__main__":
    asyncio.run(main())
