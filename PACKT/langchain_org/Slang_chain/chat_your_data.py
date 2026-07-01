import os
import sys
from dotenv import find_dotenv, load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv(find_dotenv())
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    raise ValueError("SarvamAI_API_KEY not found in .env file")

# Sarvam uses an OpenAI-compatible API pattern in LangChain.
os.environ["OPENAI_API_KEY"] = api_key
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

print("Loading documents from data/ directory...")
documents = []

txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
documents.extend(txt_loader.load())

md_loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
documents.extend(md_loader.load())

print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks")

# Use local lexical retrieval to avoid external embeddings endpoint issues.
retriever = BM25Retriever.from_documents(splits)
retriever.k = 3

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        base_url="https://api.sarvam.ai/v1",
        model="sarvam-m",
        temperature=0,
    ),
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

print("\n" + "="*50)
print("Chat with your documents!")
print("Type 'quit', 'q', or 'exit' to stop")
print("="*50 + "\n")

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        print("Goodbye!")
        sys.exit()
    
    result = chain({"question": query, "chat_history": chat_history})
    print(f"\nAnswer: {result['answer']}\n")
"""
    # Show source documents
    if result.get("source_documents"):
        print("Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"  {i}. {source}")
        print()
    
    chat_history.append((query, result["answer"]))
    query = None
"""

