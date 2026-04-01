# rag_engine.py
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import os

EMBED_MODEL = "nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

def load_index():
    chroma_client = ChromaVectorStore.from_params(
        persist_dir="./data/chroma_storage",
        db_name="pdf_rag",
    )
    storage_context = StorageContext.from_defaults(vector_store=chroma_client)
    return VectorStoreIndex.from_vector_store(chroma_client, storage_context=storage_context)

def query_rag(query: str, top_k: int = 3):
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return str(response)

def query_by_page(page_num: int, context: str = ""):
    """Query specific page number from PDFs."""
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=1)
    query = f"Page {page_num} content"
    if context:
        query += f" Context: {context}"
    response = query_engine.query(query)
    return str(response)

def get_pdf_info():
    """Return info about indexed PDFs."""
    from llama_index.core import StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    
    chroma_client = ChromaVectorStore.from_params(
        persist_dir="./data/chroma_storage",
        db_name="pdf_rag",
    )
    storage_context = StorageContext.from_defaults(vector_store=chroma_client)
    index = VectorStoreIndex.from_vector_store(chroma_client, storage_context=storage_context)
    
    return {
        "total_nodes": index.vector_store.size(),
        "pdf_folder": "./data/pdfs",
        "embedding_model": EMBED_MODEL,
    }

if __name__ == "__main__":
    print(query_rag("What is the project architecture?"))
    print(query_by_page(1))
