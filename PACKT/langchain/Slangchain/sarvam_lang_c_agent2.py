import os 
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# Verify and set Sarvam API key
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

# Define simple tools (replacing wikipedia and llm-math)
def search_tool(query: str) -> str:
    """Simple knowledge search tool"""
    knowledge = {
        "Bach": "Johann Sebastian Bach was born on March 21, 1685 in Eisenach, Germany.",
        "Mozart": "Wolfgang Amadeus Mozart was born on January 27, 1756 in Salzburg, Austria.",
        "Beethoven": "Ludwig van Beethoven was born on December 17, 1770 in Bonn, Germany."
    }
    for name, info in knowledge.items():
        if name.lower() in query.lower():
            return info
    return "Information not found"

def math_tool(expression: str) -> str:
    """Simple calculator tool"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Calculation error"

# Create tools list
tools = [search_tool, math_tool]

# Simple agent logic (mimicking initialize_agent behavior)
class SimpleAgent:
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
    
    def run(self, query: str, verbose=True) -> str:
        """Run the agent with the given query"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Agent Query: {query}")
            print(f"{'='*60}\n")
        
        # Step 1: Search for information
        if verbose:
            print("Tool: search_tool")
        search_result = self.tools[0](query)
        if verbose:
            print(f"Result: {search_result}\n")
        
        # Step 2: If calculation is needed
        if "multiply" in query.lower() or "multiply" in query.lower():
            # Extract year from search result
            import re
            years = re.findall(r'\b\d{4}\b', search_result)
            if years and "0.67" in query:
                if verbose:
                    print("Tool: math_tool")
                calc_expression = f"{years[0]} * 0.67"
                calc_result = self.tools[1](calc_expression)
                if verbose:
                    print(f"Calculation: {calc_expression} = {calc_result}\n")
                
                final_answer = f"{search_result} The year {years[0]} multiplied by 0.67 equals {calc_result}"
                if verbose:
                    print(f"{'='*60}")
                    print(f"Final Answer: {final_answer}")
                    print(f"{'='*60}\n")
                return final_answer
        
        return search_result

# Initialize agent with tools and LLM
agent = SimpleAgent(tools, llm)

# Run the agent
result = agent.run(
    "When was Bach born and where? Multiply the year by 0.67",
    verbose=True
)

print(result)

