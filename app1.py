import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index)

# # Load from Streamlit secrets
# pinecone_api_key = st.secrets["PINECONE_API_KEY"]
# pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]
# pinecone_index = st.secrets["PINECONE_INDEX_NAME"]
# openai_api_key = st.secrets["OPENAI_API_KEY"]



pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index)

# 🧠 Set up LLM and Embeddings
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Create VectorStore
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.title("Pinecone + OpenAI RAG App")
query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    answer = qa_chain.run(query)
    st.write("### Answer")
    st.write(answer)
