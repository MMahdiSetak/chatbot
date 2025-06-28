import streamlit as st
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any
import time

# Import required libraries for RAG
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import chromadb

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with Document Upload",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-online {
        background-color: #2ca02c;
    }

    .status-offline {
        background-color: #d62728;
    }

    .status-warning {
        background-color: #ff7f0e;
    }

    .document-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f9f9f9;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }

    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #388e3c;
    }

    .source-citation {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }

    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d32f2f;
    }

    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


class RAGChatbotSystem:
    def __init__(self):
        self.initialize_session_state()
        # self.vector_store = None
        self.conversation_chain = None

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "documents" not in st.session_state:
            st.session_state.documents = []
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        if "vector_store_ready" not in st.session_state:
            st.session_state.vector_store_ready = False
        if "ollama_status" not in st.session_state:
            st.session_state.ollama_status = "checking"
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

    def check_ollama_status(self):
        """Check if Ollama is running and available"""
        try:
            # Test connection to Ollama
            test_llm = ChatOllama(
                model=st.session_state.selected_model,
                base_url="http://localhost:11434"
            )
            # Simple test query
            test_llm.invoke("test")
            st.session_state.ollama_status = "online"
            return True
        except Exception as e:
            st.session_state.ollama_status = "offline"
            return False

    def get_status_indicator(self, status: str) -> str:
        """Return HTML for status indicator"""
        if status == "online":
            return '<span class="status-indicator status-online"></span>Online'
        elif status == "offline":
            return '<span class="status-indicator status-offline"></span>Offline'
        else:
            return '<span class="status-indicator status-warning"></span>Checking...'

    def process_uploaded_file(self, uploaded_file, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Process uploaded file and extract text"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Choose appropriate loader based on file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == 'csv':
                loader = CSVLoader(tmp_file_path)
            elif file_extension == 'md':
                loader = UnstructuredMarkdownLoader(tmp_file_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Load and split documents
            documents = loader.load()

            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            split_docs = text_splitter.split_documents(documents)

            # Add metadata
            for doc in split_docs:
                doc.metadata.update({
                    "source_file": uploaded_file.name,
                    "file_type": file_extension,
                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
                })

            # Clean up temporary file
            os.unlink(tmp_file_path)

            return split_docs

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None

    def setup_vector_store(self, documents: List):
        """Setup or update ChromaDB vector store"""
        try:
            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                model=st.session_state.get("embedding_model", "nomic-embed-text"),
                base_url="http://localhost:11434"
            )

            # Create or update Chroma vector store
            if st.session_state.vector_store is None:
                # Create new vector store
                st.session_state.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory="./chroma_db",
                    collection_name="rag_documents",

                )
            else:
                # Add documents to existing vector store
                st.session_state.vector_store.add_documents(documents)

            st.session_state.vector_store_ready = True
            return True

        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
            return False

    def setup_conversation_chain(self):
        """Setup conversational retrieval chain"""
        try:
            if not st.session_state.vector_store_ready or st.session_state.vector_store is None:
                return False

            # Initialize LLM
            llm = ChatOllama(
                model=st.session_state.get("selected_model", "llama3.1:8b"),
                base_url="http://localhost:11434",
                temperature=st.session_state.get("temperature", 0.1)
            )

            # Create retriever
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": st.session_state.get("retrieval_k", 5)}
            )

            # Custom prompt template
            custom_prompt = PromptTemplate(
                template="""Use the following context to answer the question. If you cannot answer the question based on the context, say so clearly.

Context: {context}

Chat History: {chat_history}

Question: {question}

Please provide a detailed answer based on the context provided. If you reference specific information, mention which document it comes from.

Answer:""",
                input_variables=["context", "chat_history", "question"]
            )

            # Create conversation chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.conversation_memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt}
            )

            return True

        except Exception as e:
            st.error(f"Error setting up conversation chain: {str(e)}")
            return False

    def generate_response(self, query: str):
        """Generate response using RAG pipeline or general chat"""
        try:
            # Check if we have documents for RAG
            if st.session_state.vector_store_ready and st.session_state.vector_store is not None:
                return self.generate_rag_response(query)
            else:
                return self.generate_general_response(query)
        except Exception as e:
            return f"Error generating response: {str(e)}", []

    def generate_rag_response(self, query: str):
        """Generate response using RAG pipeline with documents"""
        try:
            if self.conversation_chain is None:
                if not self.setup_conversation_chain():
                    # Fallback to general chat if RAG setup fails
                    return self.generate_general_response(query)

            with st.spinner("Searching documents and thinking..."):
                result = self.conversation_chain.invoke({"question": query})
                answer = result.get("answer", "Sorry, I couldn't generate a response.")
                source_documents = result.get("source_documents", [])
                return answer, source_documents
        except Exception as e:
            # Fallback to general chat on error
            st.warning("RAG system error, falling back to general chat.")
            return self.generate_general_response(query)

    def generate_general_response(self, query: str):
        """Generate general conversation response without documents"""
        try:
            # Initialize general chat LLM
            llm = ChatOllama(
                model=st.session_state.get("selected_model", "llama3.1:8b"),
                base_url="http://localhost:11434",
                temperature=st.session_state.get("temperature", 0.7)  # Higher temp for general chat
            )

            # Create conversation context from chat history
            chat_history = ""
            for msg in st.session_state.messages[-6:]:  # Last 6 messages for context
                role = "Human" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n"

            # Enhanced prompt for general conversation
            prompt = f"""Engage in natural conversation using the context below. Respond to the user's message with exactly ONE complete answer. 
        Do not continue the conversation beyond this response.
            Recent conversation:
            {chat_history}
            Human: {query}
            Assistant:"""
            response = llm.invoke(prompt)

            # Return response and empty sources list
            return response.content, []

        except Exception as e:
            # Fallback response on error
            error_msg = f"⚠️ Error in general chat: {str(e)}"
            return error_msg, []

    def render_sidebar(self):
        """Render sidebar with document upload and configuration"""
        st.sidebar.title("📚 Document Management")

        # System Status
        st.sidebar.subheader("🔧 System Status")

        # Check Ollama status
        ollama_running = self.check_ollama_status()
        st.sidebar.markdown(
            f"**Ollama:** {self.get_status_indicator(st.session_state.ollama_status)}",
            unsafe_allow_html=True
        )

        # Vector DB status
        vector_status = "online" if st.session_state.vector_store_ready else "offline"
        st.sidebar.markdown(
            f"**Vector DB:** {self.get_status_indicator(vector_status)}",
            unsafe_allow_html=True
        )

        # Documents status
        doc_count = len(st.session_state.documents)
        doc_status = "online" if doc_count > 0 else "warning"
        st.sidebar.markdown(
            f"**Documents:** {self.get_status_indicator(doc_status)} {doc_count} loaded",
            unsafe_allow_html=True
        )

        st.sidebar.divider()

        # File Upload Section
        st.sidebar.subheader("📄 Upload Documents")

        uploaded_files = st.sidebar.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'md', 'csv'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, Markdown, or CSV files"
        )

        if uploaded_files and not st.session_state.processing:
            if st.sidebar.button("🔄 Process Documents", type="primary"):
                st.session_state.processing = True
                self.process_documents(uploaded_files)
                st.session_state.processing = False
                st.rerun()
                # st.experimental_rerun()

        # Configuration Section
        st.sidebar.subheader("⚙️ Configuration")

        # Model Selection
        st.session_state.selected_model = st.sidebar.selectbox(
            "LLM Model",
            ["llama3.1:8b", "tinyllama:1.1b-chat-v1-q4_K_M", "mistral", "phi3"],
            index=0
        )

        st.session_state.embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["nomic-embed-text", "mxbai-embed-large"],
            index=0
        )

        # Advanced Settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1
            )

            st.session_state.chunk_size = st.selectbox(
                "Chunk Size",
                [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
                index=9
            )

            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=500,
                value=50
            )

            st.session_state.retrieval_k = st.selectbox(
                "Documents to Retrieve",
                [3, 5, 7, 10, 15, 20, 30, 50],
                index=6
            )

        # Document Management
        if st.session_state.documents:
            st.sidebar.subheader("📋 Uploaded Documents")

            for i, doc_info in enumerate(st.session_state.documents):
                with st.sidebar.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {doc_info['name']}")
                        st.caption(f"Type: {doc_info['type']} | Chunks: {doc_info['chunks']}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                            self.delete_document(i)
                            st.experimental_rerun()

        # Clear Chat Button
        st.sidebar.divider()
        if st.sidebar.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.experimental_rerun()

    def process_documents(self, uploaded_files):
        """Process uploaded documents"""
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        all_documents = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")

            # Process file
            documents = self.process_uploaded_file(
                uploaded_file,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )

            if documents:
                all_documents.extend(documents)

                # Add to session state
                doc_info = {
                    "name": uploaded_file.name,
                    "type": uploaded_file.name.split('.')[-1].upper(),
                    "size": len(uploaded_file.getvalue()),
                    "chunks": len(documents),
                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.documents.append(doc_info)

        # Setup vector store with all documents
        if all_documents:
            status_text.text("Setting up vector store...")
            if self.setup_vector_store(all_documents):
                status_text.text("✅ Documents processed successfully!")
                st.sidebar.success(f"Processed {len(uploaded_files)} documents with {len(all_documents)} chunks")
            else:
                status_text.text("❌ Error setting up vector store")

        progress_bar.empty()
        status_text.empty()

    def delete_document(self, index: int):
        """Delete a document from the collection"""
        if 0 <= index < len(st.session_state.documents):
            doc_info = st.session_state.documents[index]
            st.session_state.documents.pop(index)
            st.sidebar.success(f"Deleted {doc_info['name']}")

            # Note: In a production system, you would also remove from vector store
            # For now, we'll just remove from the display list

    def render_chat_interface(self):
        """Render main chat interface"""
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Chat mode indicator
        if st.session_state.vector_store_ready:
            st.info("🤖 **RAG Mode**: I can answer questions using your uploaded documents and general knowledge.")
        else:
            st.info("💬 **General Chat Mode**: I'm ready to chat! Upload documents to enable RAG capabilities.")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"**{source['source_file']}** (Type: {source['file_type']})")
                            st.write(f"Content: {source['content'][:200]}...")

        # Chat input
        # if query := st.chat_input("Ask a question about your documents..."):
        if query := st.chat_input("Type your message here..."):
            # if not st.session_state.vector_store_ready:
            #     st.error("Please upload and process documents first before asking questions.")
            #     return

            if not self.check_ollama_status():
                st.error("Ollama is not running. Please start Ollama first.")
                return

            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})

            # Display user message
            with st.chat_message("user"):
                st.write(query)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response, sources = self.generate_response(query)

                st.write(response)

                # Format sources
                source_info = []
                if sources:
                    with st.expander("📚 Sources"):
                        for doc in sources:
                            metadata = doc.metadata
                            source_info.append({
                                "source_file": metadata.get("source_file", "Unknown"),
                                "file_type": metadata.get("file_type", "Unknown"),
                                "content": doc.page_content
                            })

                            st.write(
                                f"**{metadata.get('source_file', 'Unknown')}** (Type: {metadata.get('file_type', 'Unknown')})")
                            st.write(f"Content: {doc.page_content[:200]}...")

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": source_info
                })

    def run(self):
        """Main application entry point"""
        # Render sidebar
        self.render_sidebar()

        # Render main chat interface
        self.render_chat_interface()

        # Show helpful information if no documents are uploaded
        if not st.session_state.documents:
            st.info("""
            👋 **Welcome to the RAG Chatbot!**

            To get started:
            1. Upload some documents using the sidebar (PDF, TXT, MD, or CSV files)
            2. Click "Process Documents" to index them
            3. Start asking questions about your documents!

            Make sure Ollama is running with the required models:
            - `ollama pull llama3.1`
            - `ollama pull nomic-embed-text`
            """)


# Main application
if __name__ == "__main__":
    # Initialize and run the RAG chatbot system
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    os.environ["OLLAMA_GPU_LAYER"] = "cuda"
    chatbot = RAGChatbotSystem()
    chatbot.run()
