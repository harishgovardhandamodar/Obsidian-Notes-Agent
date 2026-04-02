# rag_engine.py (Corrected)
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import os

EMBED_MODEL = "nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

CHROMA_PERSIST_DIR = "./data/chroma_storage"
CHROMA_COLLECTION_NAME = "pdf_rag"

def load_index():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_client = ChromaVectorStore(
        chroma_client=client,
        collection_name=CHROMA_COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=chroma_client)
    return VectorStoreIndex.from_vector_store(chroma_client, storage_context=storage_context)


def query_rag(query: str, top_k: int = 3):
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return str(response)

def query_by_page(page_num: int, context: str = ""):
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=1)
    query = f"Page {page_num} content"
    if context:
        query += f" Context: {context}"
    response = query_engine.query(query)
    return str(response)

def get_pdf_info():
    chroma_client = ChromaVectorStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHMA_COLLECTION_NAME,
    )
    return {
        "total_nodes": chroma_client.count(),
        "pdf_folder": "./data/pdfs",
        "embedding_model": EMBED_MODEL,
    }

if __name__ == "__main__":
    print(query_rag("What is the project architecture?"))
    print(query_by_page(1))
