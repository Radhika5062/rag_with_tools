from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

file = "attention.pdf"

# Read the pdf
loader = PyPDFLoader(file)
documents = loader.load()

# Split the data to match the context size
textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
documents = textSplitter.split_documents(documents=documents)

# Embeddings
embeddings = OllamaEmbeddings(model="llama3")

# db
db = FAISS.from_documents(documents=documents[:30], embedding=embeddings)

# create prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following on the basis of provided context. Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>
    Question: {input}
"""
)

# create llm
llm = OllamaLLM(model = 'llama2')

# create stuff document chanins
document_chains = create_stuff_documents_chain(llm=llm, prompt=prompt)

# create retriever
retriever = db.as_retriever()

# create retrieval chain
retriever_chain = create_retrieval_chain(retriever, document_chains)

st.title("RAG demo")
st.markdown("This demo uses PDF loader from the Langchain library to read the attention is all you need document \
             stored in local directory to answer user queries. These requests are logged in Langsmith")
input_text = st.text_input("What would you like to search?")

if input_text:
    response = retriever_chain.invoke({"input": input_text})
    st.write(response['answer'])
