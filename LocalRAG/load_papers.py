# ingest_pipeline.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load PDFs
loader = PyMuPDFLoader("papers/")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)

# Embed
embeddings = HuggingFaceEmbeddings(model_name="nomic-embed-text")
vector_store = Chroma.from_documents(chunks, embeddings)

# Save for retrieval
vector_store.persist()