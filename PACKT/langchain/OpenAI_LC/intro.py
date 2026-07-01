from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Load environment variables from .env file (override existing)
load_dotenv(override=True)

# Requires OPENAI_API_KEY in the environment.
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.5,
    timeout=10,
    max_tokens=200,
)

response = model.invoke(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
)

print(response.content)