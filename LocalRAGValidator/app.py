import streamlit as st
import requests
import json

st.title("🔍 Local RAG Visualizer")

# Sidebar for Settings
query = st.sidebar.text_input("Ask your RAG system:")
if st.sidebar.button("Visualize Retrieval"):
    with st.spinner("Searching local context..."):
        # Call your local Python script via subprocess or API
        # For simplicity, assume we call a local API endpoint
        # In production, run rag_engine.py as a FastAPI server
        
        st.info("Simulating RAG Response...")
        st.write("Here is the retrieved context:")
        st.code("Vector Score: 0.98 \n Source: architecture.md", language="json")
        
        st.write("Model Response:")
        st.markdown("Based on the retrieved architecture, the system uses...")

# Visualization of Vector DB
st.subheader("Vector Store Status")
st.metric(label="Total Documents", value=142)
st.metric(label="Last Updated", value="Just now")
