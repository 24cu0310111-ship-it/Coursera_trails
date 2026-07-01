import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def search_tool(query):
    knowledge = {
        "Bach": "Born March 21, 1685 in Eisenach, Germany.",
        "Mozart": "Born January 27, 1756 in Salzburg, Austria.",
    }
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

    def run(self, query, verbose=True):
        if verbose:
            print(f"\nQuery: {query}\n")
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

try:
    with open("./test.md") as f:
        print("File:", f.read()[:100])
except:
    print("File not found")
