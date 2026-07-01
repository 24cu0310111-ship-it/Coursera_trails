from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a short bedtime story about a unicorn."}
    ]
)

print(response.choices[0].message.content)
