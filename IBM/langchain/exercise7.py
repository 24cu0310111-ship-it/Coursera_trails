from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL

# Create Python REPL once
python_repl = PythonREPL()

# -------------------------------
# Calculator Tool
# -------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.

    Input should be a valid mathematical expression.
    Example:
    25 + 63
    15 * 7
    100 / 4
    """
    try:
        result = python_repl.run(f"print({expression})")
        return result.strip()
    except Exception as e:
        return f"Error calculating: {e}"


# -------------------------------
# Text Formatter Tool
# -------------------------------
@tool
def format_text(text: str) -> str:
    """
    Format text.

    Input format:
    uppercase: hello world
    lowercase: HELLO WORLD
    titlecase: langchain is awesome
    """

    try:
        format_type, value = text.split(":", 1)

        format_type = format_type.strip().lower()
        value = value.strip()

        if format_type == "uppercase":
            return value.upper()

        elif format_type == "lowercase":
            return value.lower()

        elif format_type == "titlecase":
            return value.title()

        else:
            return "Invalid format type. Use uppercase, lowercase or titlecase."

    except Exception as e:
        return f"Error formatting text: {e}"


# -------------------------------
# Tools
# -------------------------------
tools = [calculator, format_text]


# -------------------------------
# Prompt
# -------------------------------
prompt_template = """
You are a helpful assistant.

You have access to the following tools:

{tools}

Tool names:
{tool_names}

Use the tools whenever necessary.

Question:
{input}

Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)


# -------------------------------
# Agent
# -------------------------------
agent = create_react_agent(
    llm=llama_llm,
    tools=tools,
    prompt=prompt
)


# -------------------------------
# Agent Executor
# -------------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


# -------------------------------
# Test Questions
# -------------------------------
test_questions = [
    "What is 25 + 63?",
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7",
    "titlecase: langchain is awesome"
]


# -------------------------------
# Run Tests
# -------------------------------
for question in test_questions:

    print("\n==============================")
    print("Question:", question)

    result = agent_executor.invoke(
        {
            "input": question
        }
    )

    print("\nFinal Answer:")
    print(result["output"])