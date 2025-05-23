# phase 1 imported
import streamlit as st

# phase 2 imported
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

st.title("RAG Chatbot!")

# Setup a session state variable to hold all the old messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat messages
for messages in st.session_state.messages:
    st.chat_message(messages["role"]).markdown(messages["content"])

# User input
prompt = st.chat_input("Ask me anything about the document")

if prompt:
    st.chat_message("user").markdown(prompt)
    # Add the user message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare system prompt
    groq_sys_prompt = ChatPromptTemplate.from_template("""
    You are very smart at everything, you always give the best, the most accurate and most precise answers.
    Answer the following Question: {user_prompt}.
    Start the answer directly. No small talk please.
    """)

    # Create a chat groq llama model
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = "llama3-8b-8192"
    
    groq_chat = ChatGroq(
        api_key=groq_api_key,
        model_name=model
    )

    # Build the chain
    chain = groq_sys_prompt | groq_chat | StrOutputParser()

    # Run the chain
    response = chain.invoke({"user_prompt": prompt})

    # Display and store the assistant message
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
