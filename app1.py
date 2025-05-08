import streamlit as st
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone

# Load secrets (defined in Streamlit Cloud → App settings → Secrets)
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]
pinecone_index = st.secrets["PINECONE_INDEX_NAME"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index)

# Set up LLM + Embeddings
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create VectorStore
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# UI
st.title("Pinecone + OpenAI RAG App")
query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    answer = qa_chain.run(query)
    st.markdown("### Answer")
    st.write(answer)
