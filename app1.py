import requests
import streamlit as st
import os
import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone  # Updated import
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Check internet connection
try:
    response = requests.get("https://www.google.com", timeout=5)
    st.write(f"Google Status Code: {response.status_code}")
except requests.exceptions.RequestException as e:
    st.error(f"Error connecting to Google: {e}")

# Load secrets
openai_api_key = st.secrets["openai"]["api_key"]
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_env = st.secrets["pinecone"]["environment"]
pinecone_index = st.secrets["pinecone"]["index_name"]

# Init Pinecone with the updated approach
pc = pinecone.Pinecone(api_key=pinecone_api_key)

if pinecone_index not in pc.list_indexes():
    pc.create_index(pinecone_index, dimension=1536, metric="cosine")

index = pc.Index(pinecone_index)

# Set up embedding & LLM
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Upload PDF
st.title("📄 PDF QA with Pinecone + LangChain")
pdf_file = st.file_uploader("Upload your PDF", type="pdf")
query = st.text_input("Ask a question about the PDF")

if pdf_file and query and st.button("Submit"):

    # Load PDF and split text
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Store in vector DB
    vectorstore = LangchainPinecone.from_documents(docs, embeddings, index_name=pinecone_index)

    # Create retriever and chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Run QA
    answer = qa_chain.run(query)
    st.write("### 🧠 Answer:")
    st.write(answer)
