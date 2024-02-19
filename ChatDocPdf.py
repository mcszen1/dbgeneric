import streamlit as st
from io import BytesIO
from google.colab import drive
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Constants and API Keys
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your actual API key
GPT_MODEL_NAME = 'gpt-4'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Function Definitions

def load_and_split_document(uploaded_file):
    """Loads and splits the document into pages."""
    if uploaded_file is not None:
        with BytesIO(uploaded_file.getbuffer()) as pdf_file:
            loader = PyPDFLoader(pdf_file)
            return loader.load_and_split()
    return None

def split_text_into_chunks(pages, chunk_size, chunk_overlap):
    """Splits text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def create_embeddings(api_key):
    """Creates embeddings from text."""
    return OpenAIEmbeddings(openai_api_key=api_key)

def setup_vector_database(documents, embeddings):
    """Sets up a vector database for storing embeddings."""
    vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=None)
    return vectordb

def initialize_chat_model(api_key, model_name):
    """Initializes the chat model with specified AI model."""
    return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.0)

def create_retrieval_qa_chain(chat_model, vector_database):
    """Creates a retrieval QA chain combining model and database."""
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    return ConversationalRetrievalChain.from_llm(chat_model, retriever=vector_database.as_retriever(), memory=memory)

def ask_question_and_get_answer(qa_chain, question):
    """Asks a question and retrieves the answer."""
    return qa_chain({"question": question})['answer']

# Main Execution Flow

def main():
    st.title("ANALISTA DE DOCUMENTOS")

    uploaded_file = st.file_uploader("Faça o upload do seu documento", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Loading and processing the document...'):
            pages = load_and_split_document(uploaded_file)
            documents = split_text_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = create_embeddings(OPENAI_API_KEY)
            vector_database = setup_vector_database(documents, embeddings)
            chat_model = initialize_chat_model(OPENAI_API_KEY, GPT_MODEL_NAME)
            qa_chain = create_retrieval_qa_chain(chat_model, vector_database)
            st.success("Document processed successfully!")

        question = st.text_input("Enter your question about Nvidia's financial report:")
        if question:
            with st.spinner('Finding the answer...'):
                try:
                    answer = ask_question_and_get_answer(qa_chain, question)
                    st.write(f"Answer: {answer}")
                except Exception as e:
                    st.error(f"Error processing the question: {e}")

if __name__ == '__main__':
    main()