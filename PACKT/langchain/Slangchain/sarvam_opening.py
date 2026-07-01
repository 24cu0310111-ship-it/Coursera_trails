import os
from dotenv import load_dotenv, find_dotenv
from sarvamai import SarvamAI

load_dotenv(find_dotenv())

client = SarvamAI(
    api_subscription_key=os.getenv("SarvamAI_API_KEY")
)

response = client.chat.completions(
    messages=[
        {"role": "user", "content": "What is the capital of India? write a essay on it."}
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
