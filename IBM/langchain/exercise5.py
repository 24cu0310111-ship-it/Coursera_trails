from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# 1. Set up the language model
model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
}
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
project_id = "skills-network"

# Initialize the model
model =ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id,
)

llm = WatsonxLLM(model)

# 2. Create a simple conversation with chat history
"""history =ChatMessageHistory()

# Add some initial messages (optional)
history.add_user_message("Hello, my name is Alice.")
history.add_ai_message("Hi alice, how can i help you?")
##TODO: Add an AI response
print(history.messages)
ai_response=llm.invoke(history.messages)
print(ai_response)
"""
# 3. Print the current conversation history
##TODO: Print the current messages in history

# 4. Set up a conversation chain with memory
memory = ConversationBufferMemory()
conversation =ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# 5. Function to simulate a conversation
def chat_simulation(conversation, inputs):
    """Run a series of inputs through the conversation chain and display responses"""
    print("\n=== Beginning Chat Simulation ===")
    
    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"Human: {user_input}")
        
        # Get response from the conversation chain
        response = conversation.invoke(input=user_input)
        
        # Print the AI's response
        print(f"AI: {response['response']}")
    
    print("\n=== End of Chat Simulation ===")

# 6. Test with a series of related questions
test_inputs = [
    "My favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "What activities would you recommend for me?",
    "What was my favorite color again?",
    "Can you remember both my name and my favorite color?"
]

obj=chat_simulation(conversation, test_inputs)
history =ChatMessageHistory(obj)

# 7. Examine the conversation memory
print(f"\nFinal Memory Contents:{history}")
##TODO: Print the contents of the conversation memory

# 8. Create a new conversation with a different type of memory (optional)
# Try implementing ConversationSummaryMemory or another type of memory

memory_optional= ConversationSummaryMemory()
conversation_short =ConversationChain(
    llm=llm,
    memory=memory_optional,
    verbose=False,
)

def chat_simulation_summary(conversation, inputs):
    """Run a series of inputs through the conversation chain and display responses"""
    print("\n=== Beginning Chat Simulation ===")
    
    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"Human: {user_input}")
        
        # Get response from the conversation chain
        response = conversation_short.invoke(input=user_input)
        
        # Print the AI's response
        print(f"AI: {response['response']}")
    
    print("\n=== End of Chat Simulation ===")

obj1=chat_simulation_summary(conversation_short, test_inputs)
history_summary=ChatMessageHistory(obj1)

# Examine the conversation memory
print(f"\nFinal Memory Contents:{history_summary}")