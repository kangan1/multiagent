import os
import streamlit as st

from llama_index.core import SimpleDirectoryReader

from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAI

# from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain_core.documents import Document

import weaviate
import os
os.environ["OPENAI_API_KEY"] = "ysk-proj-JrxvFIX14nNDVpxAyyU-LzhEOQRc6jDIUOBH3QQg728mbkasdFdgRC802JNRYXHnYVgz0pd2B3T3BlbkFJHJOpy9eZgQAoes-kKrvrVN-muDoVHNIQgJTNUNY37kkKyrOflHzCJO4hQtOhYQTzU6oQ5d8yoA"
# Set your OpenAI key via environment variable or directly (not recommended)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Load documents
documents = SimpleDirectoryReader('E:/practice/multi_agent/data').load_data()

# Convert LlamaIndex documents to LangChain format
lc_documents = [Document(page_content=doc.text) for doc in documents]

# Initialize OpenAI LLM and Embeddings
llm = OpenAI()
embedding = OpenAIEmbeddings()

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")

# Create vector store in Weaviate
vectorstore = Weaviate.from_documents(
    documents=lc_documents,
    embedding=embedding,
    client=client
)

# Set up RetrievalQA Chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define dummy legal and medical tools for demo
def legal_agent_run(query):
    return qa_chain.run(f"Legal perspective: {query}")

def medical_agent_run(query):
    return qa_chain.run(f"Medical perspective: {query}")

legal_tool = Tool(name="LegalAgent", func=legal_agent_run, description="Handles legal queries.")
medical_tool = Tool(name="MedicalAgent", func=medical_agent_run, description="Handles medical queries.")

# Initialize multi-agent
agents = initialize_agent(
    tools=[legal_tool, medical_tool],
    llm=llm,
    agent_type="zero-shot-react-description"
)

# Streamlit UI
st.title("Multi-Agent RAG System")
user_query = st.text_input("Enter your query:")

if st.button("Submit") and user_query:
    with st.spinner("Processing..."):
        response = agents.run(user_query)
        st.write(response)
