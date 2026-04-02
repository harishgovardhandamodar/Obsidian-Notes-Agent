from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Setup Retrievers
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Graph Retriever (Neo4j)
# graph_retriever = graph_retriever_factory(...)

# Prompt
prompt = hub.pull("langchain-ai/rag-prompt")

# Chain
def retrieve_and_answer(query):
    # 1. Vector Retrieval
    context = vector_retriever.invoke(query)
    
    # 2. MCP Call (Optional: Ask LLM to write a summary note back to Obsidian)
    # mcp_server.create_note(title="Summary", content="...")
    
    # 3. Generate Answer
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})

print(retrieve_and_answer("How does the Transformer architecture work?"))