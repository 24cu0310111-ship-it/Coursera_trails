from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# TODO: Create a simple calculator tool
def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        # HINT: Use Python's eval() function for simple calculations
        # Your code here
        pass
    except Exception as e:
        return f"Error calculating: {str(e)}"

# TODO: Create a text formatting tool
def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: '[format_type]: [text]'
    where format_type is 'uppercase', 'lowercase', or 'titlecase'."""
    try:
        # HINT: Parse the input to get format type and text
        # Your code here
        pass
    except Exception as e:
        return f"Error formatting text: {str(e)}"

# TODO: Create Tool objects for our functions
# HINT: Use the Tool class to wrap the functions
tools = [
    # Your code here
]

# TODO: Create a simple prompt template
prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.
You have access to these tools:

# TODO: Fill in the rest of the prompt template
"""

# TODO: Create the agent and executor
prompt = PromptTemplate.from_template(prompt_template)
agent = # Your code here
agent_executor = # Your code here

# Test with simple questions
test_questions = [
    "What is 25 + 63?", 
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7", 
    "titlecase: langchain is awesome",
]

# TODO: Run the tests
for question in test_questions:
    print(f"\n===== Testing: {question} =====")
    # Your code here