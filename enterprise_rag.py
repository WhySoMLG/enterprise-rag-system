import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv

class AdvancedRAGSystem:
    def __init__(self, persist_directory: str = "./chroma_db"):
        print("🚀 Starting AdvancedRAGSystem initialization...")
        
        # Load API key from .env file automatically
        load_dotenv()
        
        if not os.environ.get("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY not found. Please check your .env file.")
            
        self.persist_directory = persist_directory
        
        print("⏳ Initializing Embedding Model & LLM...")
        self.embeddings = MistralAIEmbeddings(model="mistral-embed")
        self.llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
        self.store = {}
        self.vector_store = None
        self.rag_chain = None
        
        print(f"🔍 Checking for existing vector database...")
        if os.path.exists(self.persist_directory):
            # BUGFIX: If the directory exists but is empty (from a previous crash), delete it!
            if not os.listdir(self.persist_directory):
                print("🧹 Found empty ghost database folder. Cleaning up...")
                shutil.rmtree(self.persist_directory)
            else:
                print("📦 Found existing database. Loading...")
                self._load_existing_db()
        else:
            print("✨ No existing database found. System ready for document ingestion.")

    def ingest_documents(self, directory_path: str):
        print(f"Loading documents from {directory_path}...")
        
        # OCR IS NOW ENABLED: extract_images=True
        # This allows reading of scanned documents and image-based PDFs!
        loader = DirectoryLoader(
            directory_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            loader_kwargs={"extract_images": True}
        )
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found in the uploaded files.")

        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        
        # Strip out completely blank chunks
        valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        
        if not valid_chunks:
            raise ValueError("No readable text found, even with OCR. The PDF might be completely blank.")

        print("Creating Vector Store...")
        # BUGFIX: To prevent SQLite "readonly database" errors, we must empty the collection 
        # gracefully instead of deleting the folder while the database is actively connected.
        if self.vector_store is not None:
            print("🧹 Emptying existing database collection...")
            try:
                self.vector_store.delete_collection()
            except Exception:
                pass
        else:
            # Only forcefully delete the directory if the database isn't actively loaded in memory
            if os.path.exists(self.persist_directory):
                try:
                    shutil.rmtree(self.persist_directory)
                except Exception:
                    pass
            
        self.vector_store = Chroma.from_documents(
            documents=valid_chunks, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )
        print("✅ Vector store created successfully!")
        self._build_chain()

    def _load_existing_db(self):
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        self._build_chain()

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        # BUGFIX: We explicitly check for `None` so an empty database doesn't crash the system.
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Ingest documents first.")

        retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, formulate a standalone question. "
            "Do NOT answer the question, just reformulate it if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()

        def contextualized_question(input_dict: dict):
            if input_dict.get("chat_history"):
                return contextualize_q_chain
            else:
                return input_dict["question"]

        qa_system_prompt = (
            "You are an intelligent assistant. Use the retrieved context to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n\nContext:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        rag_chain = (
            RunnablePassthrough.assign(context=contextualized_question | retriever | self._format_docs)
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        self.rag_chain = RunnableWithMessageHistory(
            rag_chain, self._get_session_history, input_messages_key="question", history_messages_key="chat_history",
        )

    def chat(self, user_input: str, session_id: str = "web_session") -> str:
        if self.rag_chain is None:
            return "Please ingest some documents first before chatting."
        return self.rag_chain.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )