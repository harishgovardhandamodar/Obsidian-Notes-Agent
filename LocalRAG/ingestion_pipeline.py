import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.llms import Ollama

# Configuration
OBSIDIAN_VAULT_PATH = r"vault/"
PDF_FOLDER = r"papers/"

# 1. Setup Local LLM & Embeddings
llm = Ollama(model="llama3:8b")
embeddings = HuggingFaceEmbeddings(model_name="nomic-embed-text")

# 2. Load PDFs
def ingest_papers():
    loader = PyMuPDFLoader(PDF_FOLDER)
    docs = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # Save to Vector Store
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vector_store.persist()
    return chunks

# 3. Extract Entities for Knowledge Graph
def build_kg(chunks):
    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
    
    # Use LangChain to extract entities/relations from text
    # This creates Cypher queries to populate the graph
    # Note: Requires a specific extraction prompt
    extraction_prompt = """
    Extract entities and relationships from this text. 
    Format: (Entity)-[RELATION]->(Entity)
    Text: {text}
    """
    
    # Simplified example: In a real scenario, use a GraphRAG chain
    # graph.add_graph_from_documents(chunks) 
    # For now, we assume we write raw data to Obsidian first
    return graph

if __name__ == "__main__":
    chunks = ingest_papers()
    # build_kg(chunks) # Uncomment when Neo4j is running
    print("Ingestion complete.")