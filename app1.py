import streamlit as st
from llama_index.core import SimpleDirectoryReader

from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings, OpenAI

import weaviate

# Load OpenAI key securely
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Init LLM and embedding
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.7)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Weaviate client
client = weaviate.Client(
    url="http://localhost:8080"
)

# Connect to Weaviate vector store (assumes index already exists)
vectorstore = Weaviate(client=client, index_name="Documents", embedding=embedding)

# Setup RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Dummy agents (replace `.run` with actual logic if needed)
class LegalAgent:
    def run(self, q): return f"Legal insight on: {q}"

class MedicalAgent:
    def run(self, q): return f"Medical insight on: {q}"

legal_agent = LegalAgent()
medical_agent = MedicalAgent()

legal_tool = Tool(name="LegalAgent", func=legal_agent.run, description="Handles legal queries.")
medical_tool = Tool(name="MedicalAgent", func=medical_agent.run, description="Handles medical queries.")

agents = initialize_agent(
    tools=[legal_tool, medical_tool],
    llm=llm,
    agent_type="zero-shot-react-description"
)

# Streamlit UI
st.title("🧠 Multi-Agent RAG System")
user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    response = agents.run(user_query)
    st.write(response)
