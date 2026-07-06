from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.chains import RetrievalQA

# 1. Load a document about AI
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
documents =loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. Set up the embedding model. (Use an embedding model to create vector representations.)
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    
}

embedding_model =WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2"
    url="https://us-south.ml.cloud_ibm.com",
    project_id="skills-network",
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

# 6. Define a function to search for relevant information
def search_documents(query):
    """Search for documents relevant to a query using the retriever"""
    # LangChain's native method name is invoke()
    docs = retriever.invoke(query)
    return docs
# 7. Test with a few queries
test_queries = [
    "What is LangChain?",
    "How do retrievers work?",
    "Why is document splitting important?"
]
for query in test_queries:
    print(f"\nQUERY: '{query}'")
    print("=" * 50)
    
    results = search_documents(query)
    # Print the results
    ##TODO: Display the results clearly
if not results:
        print("No matching text pieces found.")
else:
    for idx, doc in enumerate(results, 1):
        print(f"📄 Match {idx} (Source: {doc.metadata.get('source', 'Unknown')}):")
        # Print text content, stripped of excess whitespace
        print(f"{doc.page_content.strip()[:300]}...") 
        print("-" * 40)
