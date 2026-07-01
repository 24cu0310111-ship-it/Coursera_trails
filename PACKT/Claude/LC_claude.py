# pip install -qU langchain "langchain[anthropic]"
import os

import anthropic
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def validate_anthropic_env() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "")

    if not api_key or not api_key.startswith("sk-ant-"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing or invalid. Set a real Anthropic key (starts with 'sk-ant-')."
        )

    if "localhost" in base_url or "127.0.0.1" in base_url:
        raise RuntimeError(
            "ANTHROPIC_BASE_URL points to localhost. Remove it for real Anthropic API usage:"
            " Remove-Item Env:ANTHROPIC_BASE_URL"
        )

    # Force the official Anthropic endpoint for this script.
    os.environ["ANTHROPIC_BASE_URL"] = "https://api.anthropic.com"


validate_anthropic_env()
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
try:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(response)
except anthropic.APIConnectionError as exc:
    raise RuntimeError(
        "Could not connect to Anthropic API. Check internet/firewall/proxy settings."
    ) from exc