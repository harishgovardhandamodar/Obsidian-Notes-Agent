# mcp_server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from rag_engine import query_rag # Import your RAG logic
import os

# Initialize Server
server = Server("local-codex-rag")

# Tool 1: Search Knowledge Base (RAG)
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
        )
    ]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "search_knowledge_base":
        return [TextContent(type="text", text=query_rag(arguments["query"]))]
    
    # Tool 2: Read File (Codex Capability)
    elif name == "read_file":
        path = arguments.get("path")
        if os.path.exists(path):
            with open(path, "r") as f:
                return [TextContent(type="text", text=f.read())]
        return [TextContent(type="text", text="File not found")]
    
    return []

# Run Server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, None)

if __name__ == "__main__":
    asyncio.run(main())
