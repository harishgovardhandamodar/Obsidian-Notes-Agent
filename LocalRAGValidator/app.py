# app.py
import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Local RAG Visualizer", page_icon="🔍")
st.title("🔍 Local RAG PDF Visualizer")

# Sidebar
st.sidebar.header("📄 PDF Management")
pdf_folder = "./data/pdfs"
if st.sidebar.button("📥 Ingest PDFs"):
    import subprocess
    subprocess.run(["python", "ingest_pdf.py"])
    st.success("✅ PDFs ingested successfully!")

st.sidebar.header("🔎 Query Options")
query = st.sidebar.text_input("Ask your RAG system:")
page_query = st.sidebar.number_input("Query PDF Page (1-100)", min_value=1, max_value=100, value=1)
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=3)

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 PDF Metadata")
    try:
        info = requests.get("http://localhost:8501/api/pdf-info").json()
        st.json(info)
    except:
        st.info("PDFs not indexed yet")

with col2:
    st.subheader("🔍 Retrieval Visualization")
    if st.button("Search RAG"):
        with st.spinner("Searching local context..."):
            response = requests.post("http://localhost:8501/api/search", json={"query": query, "top_k": top_k})
            if response.status_code == 200:
                result = response.json()
                st.write(result["response"])
                st.write("### Retrieved Context:")
                for i, context in enumerate(result["context"]):
                    with st.expander(f"Context {i+1}"):
                        st.code(context["text"], language="text")
            else:
                st.error("Failed to retrieve results")

# PDF Page Query
st.subheader("📄 Query Specific PDF Page")
if st.button("Query Page"):
    with st.spinner("Fetching page..."):
        response = requests.post("http://localhost:8501/api/page", json={"page": page_query})
        if response.status_code == 200:
            st.write(response.json()["response"])
