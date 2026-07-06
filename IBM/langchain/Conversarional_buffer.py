"""
REASON behind why we are using the input in the invoke method :
ConversationChain Has a Fixed Blueprint
ConversationChain is a pre-built, specialized chain. Under the hood, its default internal prompt looks something like this:
  Eg:
    Plaintext
    The following is a friendly conversation...
    Current conversation:
    {history}
    Human: {input}
    AI:

"""
#code:
# Import ConversationBufferMemory from langchain.memory module
from langchain.memory import ConversationBufferMemory

# Import ConversationChain from langchain.chains module
from langchain.chains import ConversationChain

# Create a conversation chain with the following components:
conversation = ConversationChain(
    # The language model to use for generating responses
    llm=llama_llm,
    
    # Set verbose to True to see the full prompt sent to the LLM, including memory contents
    verbose=True,
    
    # Initialize with ConversationBufferMemory that will:
    # - Store all conversation turns (user inputs and AI responses)
    # - Append the entire conversation history to each new prompt
    # - Provide context for the LLM to generate contextually relevant responses
    memory=ConversationBufferMemory()
)
conversation.invoke(input="Hello, I am a little cat. Who are you?")
conversation.invoke(input="What can you do?")
conversation.invoke(input="Who am I?.")


"""
OUTPUT:

[1m> Entering new ConversationChain chain...[0m
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question it truthfully says it does not know.

Current conversation:
Human: Hello, I am a little cat. Who are you?
AI: Hello! I am an AI assistant. It's a pleasure to meet you, little cat! How can I help you today?
Human: What can you do?
AI: I can help you answer questions, write stories, write code, or just chat with you! What would you like to do?
Human: Who am I?.
AI:

[1m> Finished chain.[0m
{'input': 'Who am I?.',
 'history': "Human: Hello, I am a little cat. Who are you?\nAI: Hello! I am an AI assistant. It's a pleasure to meet you, little cat! How can I help you today?\nHuman: What can you do?\nAI: I can help you answer questions, write stories, write code, or just chat with you! What would you like to do?",
 'response': 'You told me that you are a little cat!'}
"""