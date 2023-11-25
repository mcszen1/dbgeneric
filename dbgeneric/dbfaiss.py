import streamlit as st

import re
import time
from io import BytesIO
from typing import Any, Dict, List
import pickle

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader



def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(
        docs, OpenAIEmbeddings(openai_api_key=openai_api_key)
    )  # Create a searchable index of the chunks

    
    return index



#
#  Put a bunch a PDF files in a VectorDB using OpenAI embeddings
#

def get_index_for_pdf(pdf_files, openai_api_key):
    documents = []
    for pdf_file in pdf_files:
        text = parse_pdf(BytesIO(pdf_file))  # Extract text from the pdf
        documents = documents + text_to_docs(text)  # Divide the text up into chunks

    index = docs_to_index(documents, openai_api_key)

    return index



#
# Function to store (persist) the Faiss DB in Databutton
#
def store_index_in_db(index, name):
    faiss.write_index(index.index, "docs.index")
    # Open the file and dump to Databutton storage
    with open("docs.index", "rb") as file:
        db.storage.binary.put(f"{name}.index", file.read())
        index.index = None
        db.storage.binary.put(f"{name}.pkl", pickle.dumps(index))


#
# Function to load the Faiss DB 
#
def load_index_from_db(index_name):
    findex = db.storage.binary.get(f"{index_name}.index")

    with open("docs.index", "wb") as file:
        file.write(findex)
    index = faiss.read_index("docs.index")
    VectorDB = pickle.loads(db.storage.binary.get(f"{index_name}.pkl"))
    VectorDB.index = index

    return VectorDB
