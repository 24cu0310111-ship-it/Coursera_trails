import os 
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# Setup Sarvam API key
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    raise ValueError("SarvamAI_API_KEY not found in .env file")
os.environ["OPENAI_API_KEY"] = api_key

# Setup Sarvam AI (replacing OpenAI)
llm = ChatOpenAI(
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-m",
    temperature=0.7
)

# Define agent tools
def search_knowledge(query: str) -> str:
    """Search for historical information about famous composers and people"""
    knowledge_base = {
        "Bach": "Johann Sebastian Bach was born on March 21, 1685 in Eisenach, Germany. He was a brilliant Baroque composer.",
        "Mozart": "Wolfgang Amadeus Mozart was born on January 27, 1756 in Salzburg, Austria.",
        "Beethoven": "Ludwig van Beethoven was born on December 17, 1770 in Bonn, Germany."
    }
    
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value
    return f"No information found for '{query}'. Try searching for Bach, Mozart, or Beethoven."

def calculator(expression: str) -> str:
    """Perform mathematical calculations"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

# Tool registry for the agent
tools = {
    "search_knowledge": search_knowledge,
    "calculator": calculator
}

# Agent function that processes user queries
def agent_execute(user_query: str):
    """Execute agent logic: analyze query and use appropriate tools"""
    print("=" * 70)
    print("🤖 SARVAM AI AGENT - Advanced Tool Usage")
    print("=" * 70)
    print(f"\n📋 User Query: {user_query}\n")
    
    # Step 1: Search for information (detect if query mentions a person)
    if any(name in user_query for name in ["Bach", "Mozart", "Beethoven"]):
        print("🔍 Step 1: Searching for biographical information...")
        search_result = search_knowledge(user_query)
        print(f"   Result: {search_result}\n")
        
        # Extract year from search result
        if "1685" in search_result:
            year = 1685
        elif "1756" in search_result:
            year = 1756
        elif "1770" in search_result:
            year = 1770
        else:
            year = None
        
        # Step 2: Perform calculation if year is found
        if year and "Multiply" in user_query:
            print(f"🧮 Step 2: Calculating multiplication...")
            calc_expression = f"{year} * 0.67"
            calc_result = calculator(calc_expression)
            print(f"   {calc_result}\n")
    
    # Step 3: Generate final response
    print("✅ Agent Response Summary:")
    print("-" * 70)
    
    # Use LLM to generate a comprehensive response
    messages = [{"role": "user", "content": user_query}]
    response = llm.invoke(messages)
    print(response.content)
    print("-" * 70)

# Main execution
if __name__ == "__main__":
    query = "When was Bach born and where? Multiply the year by 0.67"
    agent_execute(query)


