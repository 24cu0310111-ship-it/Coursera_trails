import csv
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

# 1. Setup the parser and template with dynamic input variables
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="List {number} {color_type} colors.\n{format_instructions}",
    input_variables=["number", "color_type"],  # Define your variables here
    partial_variables={"format_instructions": format_instructions}
)

# 2. Read the CSV file and inject the variables
with open('data.csv', mode='r') as file:
    # DictReader maps the header names to dictionary keys automatically
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        # row is a dictionary like: {'count': 'three', 'type': 'primary'}
        
        # 3. Format the prompt dynamically with the CSV row values
        formatted_prompt = prompt_template.format(
            number=row['count'], 
            color_type=row['type']
        )
        
        print(f"--- Generated Prompt for {row['type']} colors ---")
        print(formatted_prompt)

        