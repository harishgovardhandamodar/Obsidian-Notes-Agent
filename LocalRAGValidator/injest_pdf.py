# ingest_pdf.py (Corrected)
import os
import fitz    # PyMuPDF
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings,
    SimpleDirectoryReader
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configuration
EMBED_MODEL = "nomic-embed-text"
PDF_FOLDER = "./data/pdfs"
CHROMA_PERSIST_DIR = "./data/chroma_storage"
CHROMA_COLLECTION_NAME = "pdf_rag"

#  FIX: Set Embedding Model via Settings
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

def extract_pdf_metadata(pdf_path):
    """Extract metadata from a PDF file."""
    doc = fitz.open(pdf_path)
    metadata = {
        "file_name": os.path.basename(pdf_path),
        "page_count": len(doc),
        "title": doc.metadata.get("title", "Unknown"),
        "author": doc.metadata.get("author", "Unknown"),
        "creation_date": doc.metadata.get("creationDate", "Unknown"),
    }
    return metadata

def parse_pdf_to_chunks(pdf_path):
    """Extract text and metadata from PDF."""
    metadata = extract_pdf_metadata(pdf_path)
    doc = fitz.open(pdf_path)
    
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            chunks.append({
                "text": text,
                "page": page_num + 1,
                "metadata": {**metadata, "page_num": page_num + 1}
            })
    
    return chunks

def ingest_pdfs():
    """Ingest all PDFs from the folder."""
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created PDF folder: {PDF_FOLDER}")
        return

    # ✅ Create PersistentClient explicitly
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # ✅ Initialize ChromaVectorStore with the client
    chroma_client = ChromaVectorStore(
        chroma_client=client,
        collection_name=CHROMA_COLLECTION_NAME
    )
    
    # ✅ Create StorageContext
    storage_context = StorageContext.from_defaults(vector_store=chroma_client)
    
    # ✅ Load and index documents
    documents = SimpleDirectoryReader(PDF_FOLDER).load_data()
    
    print(f"Loaded {len(documents)} documents from {PDF_FOLDER}")
    
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=CHROMA_PERSIST_DIR)
    
    print("✅ PDFs indexed successfully!")
    print(f"Total nodes indexed: {index.vector_store.size()}")

if __name__ == "__main__":
    ingest_pdfs()
