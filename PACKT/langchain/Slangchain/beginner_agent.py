# Simple LangChain Agent with Sarvam AI - Beginner Example

import os 
import re
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# Load API key from .env file
load_dotenv(find_dotenv())
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    raise ValueError("SarvamAI_API_KEY not found in .env file")
os.environ["OPENAI_API_KEY"] = api_key

# Setup Sarvam AI LLM
llm = ChatOpenAI(
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-m",
    temperature=0.7
)

# Tool 1: Search knowledge database
def search_info(query: str) -> str:
    """Search for information about famous composers"""
    database = {
        "Bach": "Johann Sebastian Bach was born in 1685 in Eisenach, Germany.",
        "Mozart": "Wolfgang Amadeus Mozart was born in 1756 in Salzburg, Austria.",
        "Beethoven": "Ludwig van Beethoven was born in 1770 in Bonn, Germany."
    }
    for name, info in database.items():
        if name.lower() in query.lower():
            return info
    return "Information not found"

# Tool 2: Calculator
def calculate(expression: str):
    """Perform mathematical calculations"""
    try:
        return eval(expression)
    except:
        return "Error in calculation"

# Agent: Uses tools to answer questions
def run_agent(question: str):
    """Agent that searches info and performs calculations"""
    print(f"\nQuestion: {question}\n")
    
    # Use search tool
    info = search_info(question)
    print(f"Search result: {info}")
    
    # Use calculator if needed
    if "multiply" in question.lower():
        years = re.findall(r'\d{4}', info)
        if years:
            result = calculate(f"{years[0]} * 0.67")
            print(f"Calculation: {years[0]} × 0.67 = {result}")
            return f"{info} Result: {result}"
    
    return info

# Run the agent
question = "When was Bach born and where? Multiply the year by 0.67"
answer = run_agent(question)
print(f"\nFinal Answer: {answer}")
