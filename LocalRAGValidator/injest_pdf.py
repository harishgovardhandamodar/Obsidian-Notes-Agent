# ingest_pdf.py
import os
import fitz  # PyMuPDF
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

# Configuration
EMBED_MODEL = "nomic-embed-text"
PDF_FOLDER = "./data/pdfs"

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
    
    nodes = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            chunks = parse_pdf_to_chunks(pdf_path)
            for chunk in chunks:
                nodes.append(chunk)
    
    print(f"Indexed {len(nodes)} chunks from PDFs")
    
    # Save to Chroma
    from llama_index.core import StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    
    chroma_client = ChromaVectorStore.from_params(
        persist_dir="./data/chroma_storage",
        db_name="pdf_rag",
    )
    
    storage_context = StorageContext.from_defaults(vector_store=chroma_client)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir="./data/chroma_storage")
    print("✅ PDFs indexed successfully!")

if __name__ == "__main__":
    ingest_pdfs()
