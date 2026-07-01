import os
import csv
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# Setup Sarvam AI
api_key = os.getenv("SarvamAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-m",
    temperature=0.7
)

# Load CSV document
def load_csv(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader)

csv_data = load_csv("./test_data.csv")

# Simple search and calculate
def search_and_calculate(query: str):

    for row in csv_data:
        name = row.get("Name","N/a").lower()
        year = int(row.get("Age","N/a"))
        favorite = row.get("Favorite Color", "Unknown")
        one_obj=(name,year,favorite)
        if name in query.lower():
            result = year * 0.67
            print(f"✅ {name.capitalize()} born {year}")
            print(f"📊 {year} * 0.67 = {result},he likes {favorite}")
            
            # Get LLM response
            response = llm.invoke(query)
            extra_info=llm.invoke(f"Summarize the data in the CSV file: {one_obj}")
            print(f"🤖 {response.content}")
            print(f"📊 Summary of CSV data: {extra_info.content}")
            return

if __name__ == "__main__":
    print("📄 CSV Data loaded from test_data.csv\n")
    print("Data:")
    for row in csv_data:
        print(f"  {row}")
    print()
    search_and_calculate("When was Bob born and where? Multiply the year by 0.67")
