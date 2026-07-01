import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

load_dotenv(find_dotenv())

# Setup Sarvam AI
api_key = os.getenv("SarvamAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-m",
    temperature=0.7
)

# Load document using LangChain TextLoader
loader = TextLoader("./test.md")
documents = loader.load()
doc_content = documents[0].page_content
print(doc_content[:100])  # Print first 100 characters of the document
# ...rest of code...

