"""
Document Querying with LangChain
Demonstrates: How to query documents using RAG (Retrieval Augmented Generation)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Sample documents (in real scenario, you'd load from files)
documents = [
    Document(page_content="LangChain is a framework for developing applications powered by language models.", 
             metadata={"source": "intro.txt"}),
    Document(page_content="OpenAI provides powerful APIs for natural language processing tasks.", 
             metadata={"source": "openai.txt"}),
    Document(page_content="Vector databases store embeddings and enable semantic search.", 
             metadata={"source": "vectors.txt"}),
    Document(page_content="RAG combines retrieval and generation for accurate question answering.", 
             metadata={"source": "rag.txt"})
]

print("Step 1: Splitting documents into chunks")
print("-" * 50)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splits = text_splitter.split_documents(documents)
print(f"Created {len(splits)} document chunks\n")

print("Step 2: Creating embeddings and vector store")
print("-" * 50)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
print("Vector store created\n")

print("Step 3: Setting up retriever")
print("-" * 50)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("Retriever ready\n")

print("Step 4: Creating QA chain")
print("-" * 50)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print("QA chain created\n")

print("Step 5: Querying the documents")
print("-" * 50)
query = "What is RAG?"
result = qa_chain.invoke({"query": query})

print(f"Question: {query}")
print(f"Answer: {result['result']}\n")

print("Source documents:")
for i, doc in enumerate(result['source_documents'], 1):
    print(f"{i}. {doc.metadata['source']}: {doc.page_content[:100]}...")
