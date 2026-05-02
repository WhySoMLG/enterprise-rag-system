import streamlit as st
import os
import shutil
from enterprise_rag import AdvancedRAGSystem
from dotenv import load_dotenv

# Automatically load the hidden .env file
load_dotenv()

st.set_page_config(page_title="Enterprise RAG", page_icon="📚", layout="wide")
st.title("📚 Enterprise RAG System")
st.write("Upload PDFs via the sidebar and chat with your documents.")

# ==========================================
# Sidebar: Setup & Document Upload
# ==========================================
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please add your OPENAI_API_KEY to the .env file.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing, chunking, and embedding documents..."):
                try:
                    # Initialize the system
                    st.session_state.rag_system = AdvancedRAGSystem()
                    
                    temp_dir = "temp_uploaded_docs"
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    st.session_state.rag_system.ingest_documents(temp_dir)
                    st.success("✅ Documents successfully ingested! You can now chat.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ==========================================
# Main Chat Interface
# ==========================================
# Only show chat if the system has been initialized
if "rag_system" not in st.session_state:
    st.info("👈 Please upload documents in the sidebar to begin.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.chat(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})