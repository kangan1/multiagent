import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Safely access nested secrets
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_env = st.secrets["pinecone"]["environment"]
pinecone_index_name = st.secrets["pinecone"]["index_name"]
openai_api_key = st.secrets["openai"]["api_key"]

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Create the index if it doesn't exist
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# Connect to the index
index = pc.Index(pinecone_index_name)

# Set up embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# Vector store and QA chain
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# UI
st.title("Pinecone + OpenAI RAG App")
query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    answer = qa_chain.run(query)
    st.write("### Answer")
    st.write(answer)
