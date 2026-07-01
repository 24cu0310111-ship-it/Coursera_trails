from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Ensure you import your specific LLM wrapper, for example:
# from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM


# 1. Define your structure using Pydantic
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# 2. Initialize the JSON output parser
output_parser = JsonOutputParser(pydantic_object=Joke)

# 3. Get the format instructions auto-generated from your schema
format_instructions = output_parser.get_format_instructions()

# 4. Set up the Prompt Template
prompt = PromptTemplate(
    template="Answer the user query.{format_instructions}\n{query}\n",
    input_variables={"query":"make a joke about programming"},
    partial_variables={"format_instructions": format_instructions},
)

# 5. Define your LLM (Replace this with your actual initialized LLM object)
# mixtral_llm = WatsonxLLM(model=model)

# 6. Build the LangChain Expression Language (LCEL) chain
chain = prompt | mixtral_llm | output_parser

# 7. Run the query
response = chain.invoke()

# 8. View your structured python dictionary output
print(response)
