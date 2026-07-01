import os
from dotenv import find_dotenv, load_dotenv
from openai import AuthenticationError

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = create_agent(model=llm, tools=tools)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "When was Bach born and where? Multiply the year by 0.67",
                }
            ]
        }
    )
except AuthenticationError:
    raise SystemExit(
        "OpenAI authentication failed. Check OPENAI_API_KEY in your environment/.env file and try again."
    )

if result.get("messages"):
    print(result["messages"][-1].content)
else:
    print(result)

