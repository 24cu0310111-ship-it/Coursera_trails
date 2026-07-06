import os
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Fixed import path for newer LangChain
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM # Imported to complete the QA chain

# 1. Load a document about AI
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
documents = loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. Set up the embedding model
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

embedding_model = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2",
    url="https://us-south.ml.cloud_ibm.com",
    project_id="skills-network",
    instance_id="openshift",  # Fixes the Pydantic ValidationError
    params=embed_params
)

# 4. Create a vector store
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    collection_name="exercise4"
)

# 5. Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. Implement a simple question-answering system (RetrievalQA)
# To answer questions, we need an LLM along with our retriever
llm = WatsonxLLM(
    model_id="google/flan-ul2",  # Standard instruction-following model
    url="https://us-south.ml.cloud_ibm.com",
    project_id="skills-network",
    instance_id="openshift"
)

# Build the QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 7. Test your system with at least 3 different questions
test_queries = [
    "What is LangChain?",
    "How do retrievers work?",
    "Why is document splitting important?"
]

for query in test_queries:
    print("\n" + "=" * 60)
    print(f"QUESTION: {query}")
    print("=" * 60)
    
    # Run the question through the QA system
    response = qa_chain.invoke({"query": query})
    answer = response.get("result", "No answer generated.")
    
    print(f"🤖 ANSWER:\n{answer.strip()}")
    print("-" * 60)

    