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

text_splitter =documents.text_splitter()

chunks =CharacterTextSplitter(chuck_size=2000,chuck_overlap=20,seperator="\n")



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

vector_store =Chroma(

collection_name="exercise4",embedding_function=Watsonx_embedding

)



# 5. Create a retriever

retriever =docsearch.as_retriever()



# 6. Define a function to search for relevant information

def search_documents(query, top_k=3):

  """Search for documents relevant to a query"""

  # Use the retriever to get relevant documents

  docs = retriever.get_relevant_documents(query)



  # Limit to top_k if specified

  return docs[:top_k]



  # 7. Test with a few queries

  test_queries = [

  "What is LangChain?",

  "How do retrievers work?",

  "Why is document splitting important?"

  ]



  for query in test_queries:

    print(f"\nQuery: {query}")

    results =search_documents(query)

    print(result,end="\n")

    # Print the results

    ##TODO: Display the results clearly

