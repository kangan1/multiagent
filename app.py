import streamlit as st
import pinecone
from llama_index import SimpleDirectoryReader
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool

# Step 1: Load Documents using LlamaIndex
documents = SimpleDirectoryReader('E:/practice/multi_agent/data').load_data()

# Step 2: Initialize OpenAI LLM
llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Step 3: Initialize Pinecone
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV"])
index_name = "your-index-name"

# Step 4: Create Embeddings and Vector Store
embedding = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

# Create Pinecone vector store with LangChain wrapper
vectorstore = Pinecone.from_documents(documents, embedding, index_name=index_name)

# Step 5: Set up RetrievalQA Chain
qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

# Query and retrieve answer
query = "Summarize recent case laws related to digital privacy in India."
response = qa_chain.run(query)
print(response)

# Step 6: Initialize Agents (Legal and Medical)
legal_tool = Tool(name="LegalAgent", func=legal_agent.run, description="Handles legal queries.")
medical_tool = Tool(name="MedicalAgent", func=medical_agent.run, description="Handles medical queries.")

agents = initialize_agent([legal_tool, medical_tool], llm, agent_type="zero-shot-react-description")

# Step 7: Streamlit Interface for User Input
st.title("Multi-Agent RAG System")
user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    response = agents.run(user_query)
    st.write(response)
