from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import streamlit as st
import os

import textwrap

# Load environment variables
load_dotenv(find_dotenv())

# Setup Sarvam API key
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    raise ValueError("SarvamAI_API_KEY not found in .env file")
os.environ["OPENAI_API_KEY"] = api_key


def create_db_from_ytube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    
    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    # Use local lexical retrieval to avoid external embeddings endpoint issues
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 4
    
    return retriever
 
 
def get_response_from_query(retriever, query, k=4):
    
    docs = retriever.invoke(query)
    
    docs_page_content = " ".join([d.page_content for d in docs])
    
    # temperature - control the creativity of the response
    chat = ChatOpenAI(
        base_url="https://api.sarvam.ai/v1",
        model="sarvam-m",
        temperature=0.2
    )
    
    # Template for the system message prompt
    template = """ 
        You are a helpful assistant that can answer questions about youtube videos 
        based on the video's transcript provided: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, simply say "I don't know".
        
        Your answers should be verbose and detailed.
        Additionally, you are also a well known youtube video script writer
        who's extremely talented and knows how to craft incredibly entertaining
        and valuable youtube video scripts.  You have a good sense of humor
        and know how to captivate viewers attention with great pattern-interrupt
        hooks. If asked to write a youtube video script based on 
        the transcript or the information you were given, make sure
        you do so in the voice of an expert person in that topic.
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    # Human question prompt
    human_template = "Please answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    # Construct the actual chat prompt template and pass the humand and system prompts
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    # use chain and pass the sarvam llm model 
    chain = chat_prompt | chat
    
    response = chain.invoke({"question": query, 
                             "docs": docs_page_content})
    # Extract content from AIMessage object
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)
    
    if isinstance(response_text, str):
        response_text = response_text.replace("\n","")
    return response_text, docs
    
   
   
   
def main():
  st.title("Youtube Assistant - Sarvam AI")
  
  youtube_url = st.text_input("Enter a Youtube URL:")
  
  if youtube_url:
      st.write("Loading...")
      retriever = create_db_from_ytube_video_url(youtube_url)
      
      st.write("Transcript fetched successfully!")
      
      question = st.text_input("Ask a question about the video")
      
      if st.button("Get Response") and question:
          st.write("Generating response...")
          
          response, _ = get_response_from_query(retriever, query=question)
          with st.expander("Response"):
              st.write(response)
   
if __name__ == "__main__":
    main()    
     
# Test!
# video_url = "https://youtu.be/rn6R4ncd2OU"
# retriever = create_db_from_ytube_video_url(video_url)

# query = "Give me 3 main points of the video? write me a simple video script about older penguins."
# response, docs = get_response_from_query(retriever, query)
# print(textwrap.fill(response, width=60))
