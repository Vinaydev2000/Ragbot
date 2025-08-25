import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

st.title("📄 Local RAG Bot")

query = st.text_input("Ask me anything about your PDF:")

if query:
    # Load vector DB
    db = Chroma(persist_directory="storage", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

    # Search for relevant chunks
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Generate answer with Ollama
    llm = Ollama(model="llama3:8b")
    answer = llm(f"Answer using the context:\n{context}\n\nQuestion: {query}")

    st.write("### 🤖 Answer:")
    st.write(answer)
    st.write("### 📚 Sources:", [d.metadata.get("source") for d in docs])

