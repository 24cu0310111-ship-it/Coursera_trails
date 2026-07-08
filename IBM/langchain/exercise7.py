from langchain_core.tools import Tool
import re



def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: '[format_type]: [text]'
    where format_type is 'uppercase', 'lowercase', or 'titlecase'."""
    try:
        if ":" not in text:
            raise ValueError("Input must be in format '[format_type]: [text]'")

        format_type, value = text.split(":", 1)
        format_type = format_type.strip().lower()
        value = value.strip()

        if format_type == "uppercase":
            return value.upper()
        if format_type == "lowercase":
            return value.lower()
        if format_type == "titlecase":
            return value.title()

        raise ValueError("format_type must be uppercase, lowercase, or titlecase")
    except Exception as e:
        return f"Error formatting text: {str(e)}"


tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Use for arithmetic expressions like '2 + 2' or '15 / 3'.",
    ),
    Tool(
        name="Text Formatter",
        func=format_text,
        description="Use for text formatting requests in the form 'uppercase: text', 'lowercase: text', or 'titlecase: text'.",
    ),
]

class SimpleAgentExecutor:
    def __init__(self, calculator_func, formatter_func) -> None:
        self.calculator_func = calculator_func
        self.formatter_func = formatter_func

    def invoke(self, input_data: dict) -> dict:
        question = input_data.get("messages", [{}])[-1].get("content", "")
        question_lower = question.lower().strip()

        def extract_text(default_text: str = "") -> str:
            quoted_match = re.search(r"['\"]([^'\"]+)['\"]", question)
            if quoted_match:
                return quoted_match.group(1)
            return default_text

        if any(symbol in question for symbol in ["+", "-", "*", "/"]):
            expression = question
            for prefix in ["what is", "calculate", "compute", "solve"]:
                if question_lower.startswith(prefix):
                    expression = question[len(prefix):].strip().rstrip("?")
                    break
            output = self.calculator_func(expression)
        elif "uppercase" in question_lower:
            text = extract_text(question.split("uppercase", 1)[-1].strip(" :?'") or question)
            output = self.formatter_func(f"uppercase: {text}")
        elif "lowercase" in question_lower:
            text = extract_text(question.split("lowercase", 1)[-1].strip(" :?'") or question)
            output = self.formatter_func(f"lowercase: {text}")
        elif "titlecase" in question_lower:
            text = extract_text(question.split("titlecase", 1)[-1].strip(" :?'") or question)
            output = self.formatter_func(f"titlecase: {text}")
        elif ":" in question:
            output = self.formatter_func(question)
        else:
            output = "I can help with simple math or text formatting requests."

        return {"messages": [{"role": "assistant", "content": output}]}


agent_executor = SimpleAgentExecutor(calculator, format_text)


test_questions = [
    "What is 25 + 63?",
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7",
    "titlecase: langchain is awesome",
]


for question in test_questions:
    print(f"\n===== Testing: {question} =====")
    result = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1]["content"])