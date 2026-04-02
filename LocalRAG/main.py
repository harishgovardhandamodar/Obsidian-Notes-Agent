# main.py
from ingestion_pipeline import ingest_papers
from obsidian_mcp_server import create_note_tool
from rag_query import query_agent

def main():
    # 1. Ingest
    print("Ingesting papers...")
    ingest_papers()
    
    # 2. Query (Simulated)
    print("Querying RAG...")
    answer = query_agent("What are the key findings?")
    
    # 3. Write Note via MCP
    print("Writing summary to Obsidian...")
    create_note_tool(
        filename="Research_Summary_2024", 
        content=answer,
        tags=["RAG", "Summary"]
    )
    print("Done. Check Obsidian.")

if __name__ == "__main__":
    main()