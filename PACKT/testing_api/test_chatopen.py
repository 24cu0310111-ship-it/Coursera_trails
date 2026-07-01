from langchain_openai import ChatOpenAI
import json

# Test the correct parameters
try:
    # Try method 1: OpenAI API compatible
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key="test-key"
    )
    print("✅ Method 1 (model, temperature, api_key) works")
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

try:
    # Try method 2: with base_url
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url="https://api.sarvam.ai/v1",
        api_key="test-key"
    )
    print("✅ Method 2 (model, base_url, api_key) works")
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# Get the class attributes
print("\nChatOpenAI attributes:")
print([x for x in dir(ChatOpenAI) if not x.startswith('_')])
