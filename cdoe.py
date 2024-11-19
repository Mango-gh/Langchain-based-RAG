#TEXT BASED LOADER
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import PyPDFLoader
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from langchain.vectorstores import Chroma



pdf_folder = r"C:\Users\treja\OneDrive\Documents\UB\Projects\RAGaws"
# Use glob to retrieve all PDF paths from the folder
pdf_paths = glob.glob(f"{pdf_folder}\\*.pdf")

# Extract text from all PDFs
docs = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
document = text_splitter.split_documents(docs)


persist_directory = "./chroma_db"
db = Chroma.from_documents(document, OpenAIEmbeddings(), persist_directory=persist_directory)

llm = ChatOpenAI(model='gpt-3.5-turbo')

prompt = PromptTemplate.from_template(
    """Give a concise answer to the question asked only based on the context provided. If the scope of the question is beyond the context then you can reply that you do not know. The context is <context> {context} </context>. Question: {input}""")

document_chain = create_stuff_documents_chain(llm,prompt)
retriver = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriver,document_chain)

st.title("Query Answering with Context")

query = st.text_input("Enter your query:")
if query:
    result = retrieval_chain.invoke({"input": query})
    st.write("Answer:", result['answer'])

