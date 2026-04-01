import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

# ================= CONFIGURATION =================
PDF_PATH = "./papers"  # Folder where your PDFs are stored
PERSIST_DIRECTORY = "./chroma_db" # Where to save the vector database
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Local embedding model
LLM_MODEL = "qwen2.5:14b" # Must match the model pulled in Ollama
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ================= 1. INGESTION =================
def ingest_documents():
    print(f"--- Ingesting PDFs from {PDF_PATH} ---")
    
    if not os.path.exists(PDF_PATH):
        os.makedirs(PDF_PATH)
        print(f"Created directory {PDF_PATH}. Please add PDFs there and run again.")
        return False

    pdf_files = glob.glob(os.path.join(PDF_PATH, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in the directory.")
        return False

    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)

    # Initialize Local Embeddings
    # We use sentence-transformers directly for OllamaEmbeddings compatibility or HF
    embeddings = SentenceTransformer(EMBEDDING_MODEL)
    
    # Wrap for LangChain compatibility
    from langchain_community.embeddings import HuggingFaceEmbeddings
    local_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create Vector Store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=local_embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"--- Ingestion Complete. {len(splits)} chunks stored. ---")
    return True

# ================= 2. QUERY ENGINE =================
def setup_rag_chain():
    # Load existing Vector Store
    local_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=local_embeddings
    )

    # Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

    # Initialize Local LLM (Qwen via Ollama)
    llm = OllamaLLM(model=LLM_MODEL)

    # Prompt Template for RAG
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, say that you don't know. 
    Keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Build Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ================= 3. MAIN EXECUTION =================
def main():
    # Step 1: Check if DB exists, if not, ingest
    if not os.path.exists(PERSIST_DIRECTORY):
        success = ingest_documents()
        if not success:
            return
    else:
        print("--- Vector DB found. Skipping ingestion. ---")

    # Step 2: Setup Chain
    rag_chain = setup_rag_chain()

    # Step 3: Interactive Loop
    print("\n--- Local Qwen RAG Ready (Type 'quit' to exit) ---")
    while True:
        query = input("\nUser: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        
        try:
            response = rag_chain.invoke(query)
            print(f"Qwen: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()