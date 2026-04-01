# rag_engine.py
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import get_response_synthesizer

# Configuration
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Initialize Ollama Settings
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

# 1. Ingestion (Run this once to load data)
# documents = SimpleDirectoryReader("./data").load_data()
# index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist(persist_dir="./data/chroma_storage")

# 2. Query Engine (The core logic)
def query_rag(query: str) -> str:
    # Load existing index
    index = VectorStoreIndex.load_from_persist_dir("./data/chroma_storage")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return str(response)

if __name__ == "__main__":
    print(query_rag("What is the project architecture?"))
