"""
Testing Chatbot Interactions
Demonstrates: Building and testing a conversational chatbot with memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

print("="*60)
print("Testing Chatbot Interactions with LangChain")
print("="*60 + "\n")

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Test 1: Basic conversation without memory
print("Test 1: Conversation WITHOUT Memory")
print("-" * 60)
basic_chain = ConversationChain(llm=llm, verbose=False)
response1 = basic_chain.predict(input="My name is Alice.")
print(f"User: My name is Alice.")
print(f"Bot: {response1}\n")

response2 = basic_chain.predict(input="What is my name?")
print(f"User: What is my name?")
print(f"Bot: {response2}\n")

# Test 2: Conversation WITH memory
print("Test 2: Conversation WITH Memory")
print("-" * 60)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

response1 = conversation.predict(input="My name is Bob.")
print(f"User: My name is Bob.")
print(f"Bot: {response1}\n")

response2 = conversation.predict(input="What is my name?")
print(f"User: What is my name?")
print(f"Bot: {response2}\n")

# Test 3: Custom prompt template
print("Test 3: Custom Chatbot with Specific Role")
print("-" * 60)
template = """You are a helpful Python programming tutor.
Current conversation:
{history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

tutor_memory = ConversationBufferMemory()
tutor_bot = ConversationChain(
    llm=llm,
    memory=tutor_memory,
    prompt=prompt,
    verbose=False
)

response = tutor_bot.predict(input="How do I create a list in Python?")
print(f"User: How do I create a list in Python?")
print(f"Tutor: {response}\n")

response = tutor_bot.predict(input="Can you show me an example?")
print(f"User: Can you show me an example?")
print(f"Tutor: {response}\n")

# Test 4: Viewing conversation history
print("Test 4: Viewing Conversation History")
print("-" * 60)
print("Chat history:")
print(tutor_memory.load_memory_variables({})["history"])
