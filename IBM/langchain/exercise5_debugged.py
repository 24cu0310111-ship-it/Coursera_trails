"""
Exercise 5 - Building a Chatbot with Memory using LangChain

This is a debugged version of your updated starter code.
It keeps the same step-by-step structure as the exercise but fixes the broken parts.
"""

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM


MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"
INSTANCE_ID = "openshift"


def build_llm():
    """Create a Watsonx ModelInference object and wrap it in a LangChain-compatible LLM."""
    model = ModelInference(
        model_id=MODEL_ID,
        params={
            GenParams.MAX_NEW_TOKENS: 256,
            GenParams.TEMPERATURE: 0.2,
        },
        credentials={"url": URL},
        project_id=PROJECT_ID,
    )

    return WatsonxLLM(model=model)


# 1. Set up the language model
llm = build_llm()

# 2. Create a simple conversation with chat history
history = ChatMessageHistory()
history.add_user_message("Hello, my name is Alice.")
history.add_ai_message("Hi Alice, how can I help you today?")
print("\nCurrent chat history:")
print(history.messages)

# 3. Set up a conversation chain with memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False,
)


# 4. Function to simulate a conversation
def chat_simulation(conversation, inputs):
    """Run a series of inputs through the conversation chain and display responses."""
    print("\n=== Beginning Chat Simulation ===")

    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i + 1} ---")
        print(f"Human: {user_input}")

        response = conversation.invoke(input=user_input)
        print(f"AI: {response['response']}")

    print("\n=== End of Chat Simulation ===")
    return memory


# 5. Test with a series of related questions
test_inputs = [
    "My favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "What activities would you recommend for me?",
    "What was my favorite color again?",
    "Can you remember both my name and my favorite color?",
]

chat_simulation(conversation, test_inputs)

# 6. Examine the conversation memory
print("\nFinal Buffer Memory Contents:")
print(memory.buffer)
print("\nLoaded memory variables:")
print(memory.load_memory_variables({}))#----------------------->DOUBT

# 7. Create a new conversation with a different type of memory (optional)
summary_memory = ConversationSummaryMemory(llm=llm)
conversation_short = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=False,
)


def chat_simulation_summary(conversation, inputs):
    """Run a short summary-memory demo with the same inputs."""
    print("\n=== Beginning Summary Memory Demo ===")

    for i, user_input in enumerate(inputs):
        print(f"\n--- Turn {i + 1} ---")
        print(f"Human: {user_input}")

        response = conversation.invoke(input=user_input)
        print(f"AI: {response['response']}")

    print("\n=== End of Summary Memory Demo ===")
    return summary_memory


chat_simulation_summary(conversation_short, test_inputs)

# 8. Examine the summary memory
print("\nFinal Summary Memory Contents:")
print(summary_memory.buffer)



"""
Expected Output:
Current chat history:
[HumanMessage(content='Hello, my name is Alice.'), AIMessage(content='Hi Alice, how can I help you today?')]

=== Beginning Chat Simulation ===

--- Turn 1 ---
Human: My favorite color is blue.
AI: <model-generated response>

--- Turn 2 ---
Human: I enjoy hiking in the mountains.
AI: <model-generated response>

...
Final Buffer Memory Contents:
Human: My favorite color is blue.
AI: <response>
Human: I enjoy hiking in the mountains.
AI: <response>
...

Loaded memory variables:
{'history': 'Human: ...\nAI: ...\nHuman: ...'}

=== Beginning Summary Memory Demo ===
...
Final Summary Memory Contents:
<summary text generated from the conversation>
"""