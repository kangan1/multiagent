import streamlit as st
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Load Pinecone API key and environment from Streamlit secrets
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_env = st.secrets["pinecone"]["environment"]
pinecone_index = st.secrets["pinecone"]["index_name"]  # Optionally specify your index name here

# Initialize Pinecone using the Pinecone class (not init())
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Check if the index exists
if pinecone_index not in pc.list_indexes().names():
    # If not, create a new index
    pc.create_index(
        name=pinecone_index,
        dimension=1536,  # Based on OpenAI embeddings size
        metric='cosine',  # Use cosine distance metric
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # Choose your desired region
        )
    )

# Connect to the index
index = pc.Index(pinecone_index)

# 🧠 Set up LLM and Embeddings
openai_api_key = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Create VectorStore
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("Pinecone + OpenAI RAG App")
query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    answer = qa_chain.run(query)
    st.write("### Answer")
    st.write(answer)
