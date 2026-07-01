from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

output_parser = CommaSeparatedListOutputParser()


format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables={"subject": "ice cream flavors"},
    partial_variables={"format_instructions": format_instructions},
)


chain = prompt | mixtral_llm | output_parser

result = chain.invoke({"subject": "ice cream flavors"})


"""In LangChain, partial variables allow you to initialize a prompt template with some of its required variables filled in ahead of time, 
while leaving the remaining variables to be filled in later when you actually run the chain.
It is essentially a way to pre-populate or "bake in" certain values into your prompt template. """
