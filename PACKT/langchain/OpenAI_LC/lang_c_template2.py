import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=150,
)

prompt = "Give me the best name for Twitter Handle for a programmer"

answer = llm.invoke(prompt)
print(answer.content)

# use template
prompt_handle_template = PromptTemplate(
    input_variables=["profession", "age"],
    template="Give me the best name for Twitter Handle for a {profession} who is {age} years old."
)
print(prompt_handle_template.format(profession="Astronaut", age=30))

# chain
chain = prompt_handle_template | llm
result = chain.invoke({"profession": "Dancer", "age": 25})
print(result.content)



