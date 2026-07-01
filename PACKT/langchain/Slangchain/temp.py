import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# Sarvam AI Setup
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    raise ValueError("SarvamAI_API_KEY not set")
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(base_url="https://api.sarvam.ai/v1", model="sarvam-m", temperature=0)

def search_tool(query):
    knowledge = {"Bach": "Born March 21, 1685 in Eisenach, Germany.", "Mozart": "Born January 27, 1756 in Salzburg, Austria."}
    for name, info in knowledge.items():
        if name.lower() in query.lower():
            return info
    return "Not found"

def math_tool(expr):
    try:
        return str(eval(expr))
    except:
        return "Error"

class Agent:
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm

    def run(self, query):
        print(f"\n🤖 Sarvam AI Agent\nQuery: {query}\n")
        result = self.tools[0](query)
        print(f"Search: {result}\n")
        
        if "multiply" in query.lower():
            import re
            year = re.findall(r"\d{4}", result)
            if year:
                calc = f"{year[0]} * 0.67"
                ans = self.tools[1](calc)
                print(f"Result: {calc} = {ans}\n")
                return f"{result} Answer: {ans}"
        return result

agent = Agent([search_tool, math_tool], llm)
result = agent.run("When was Bach born and where? Multiply the year by 0.67")
print(result)
