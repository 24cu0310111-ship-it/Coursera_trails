import os
import sys
from dotenv import find_dotenv, load_dotenv
import openai

# Updated imports for modern LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Loading from persisted vector store...")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    print("Loading documents from data/ directory...")
    # Load different document types
    documents = []
    
    # Load .txt files
    txt_loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    # Load .md files
    md_loader = DirectoryLoader("data/", glob="**/*.md", loader_cls=TextLoader)
    documents.extend(md_loader.load())
    
    # Load .csv files
    csv_loader = DirectoryLoader("data/", glob="**/*.csv", loader_cls=CSVLoader)
    documents.extend(csv_loader.load())
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    
    if PERSIST:
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory="persist"
        )
    else:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up the conversational retriever chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# CONVERSATIONAL LOOP - SAVES HISTORY
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
    
    # Show source documents
    if result.get("source_documents"):
        print("Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"  {i}. {source}")
        print()
    
    chat_history.append((query, result["answer"]))
    query = None