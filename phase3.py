# Phase 1: Streamlit
import streamlit as st

# Phase 2: General setup
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3: LangChain + Vectorstore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()

# Streamlit UI
st.title("📄 RAG Chatbot for PDF")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for messages in st.session_state.messages:
    st.chat_message(messages["role"]).markdown(messages["content"])

# Caching vectorstore creation
@st.cache_resource
def get_vectorstore():
    pdf_path = "./Data Management.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore

# User input
prompt = st.chat_input("Ask me anything about the document")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    
    # GROQ prompt
    groq_prompt = ChatPromptTemplate.from_template("""
    You are very smart at everything, you always give the best, the most accurate and most precise answers.
    Answer the following Question: {user_prompt}.
    Start the answer directly. No small talk please.
    """)

    # GROQ LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = "llama3-8b-8192"
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model)

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # QA chain
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = chain.invoke({"query": prompt})
        response = result["result"]

        # Show result
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")