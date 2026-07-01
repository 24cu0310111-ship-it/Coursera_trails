# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent and print the weather response
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

final_message = response["messages"][-1]

# Some chat models return content as a string, others as a list of blocks.
if isinstance(final_message.content, list):
    text = "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in final_message.content
    ).strip()
    print(text)
else:
    print(final_message.content)
