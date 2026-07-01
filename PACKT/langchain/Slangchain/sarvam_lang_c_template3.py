import os 
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv(find_dotenv())

# Setup Sarvam AI API key
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    raise ValueError("SarvamAI_API_KEY not found in .env file")
os.environ["OPENAI_API_KEY"] = api_key

# Setup Sarvam AI with LangChain (OpenAI-compatible)
llm = ChatOpenAI(
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-m",
    temperature=0.6
)

# Simple prompt without template
prompt = "Give me the best name for Twitter Handle for a programmer"

answer = llm.invoke(prompt)
print("=" * 60)
print("WITHOUT TEMPLATE:")
print("=" * 60)
print(answer.content)

# Use template with variables
print("=" * 60)
print("WITH TEMPLATE:")
print("=" * 60)

prompt_handle_template = PromptTemplate(
    input_variables=["profession", 'age'],
    template="Give me the best name for Twitter Handle for a {profession} who is {age} years old."
)

# Format template (just to show)
#formatted = prompt_handle_template.format(profession="Astronaut", age=30)
#print(f"Formatted template: {formatted}\n")

# Chain - connect LLM with template (using LCEL - modern LangChain syntax)
chain = prompt_handle_template | llm

# Run the chain
result = chain.invoke({"profession": "Dancer", "age": 25})
print("=" * 60)
print("CHAIN RESULT:")
print("=" * 60)
print(result.content)
