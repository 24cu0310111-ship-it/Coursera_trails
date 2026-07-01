import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, AuthenticationError

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.getenv("SarvamAI_API_KEY")

# Check if API key exists
if not api_key:
    print("❌ ERROR: SarvamAI_API_KEY not found in .env file")
    exit(1)

print(f"✓ API Key found: {api_key[:20]}...{api_key[-10:]}")
print("\nTesting Sarvam AI API key validity...")

try:
    # Initialize OpenAI-compatible client for Sarvam AI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.sarvam.ai/v1"  # Sarvam AI endpoint
    )
    
    # Make a simple test request
    response = client.chat.completions.create(
        model="sarvam-m",  # Options: "sarvam-m", "sarvam-30b", "sarvam-105b"
        messages=[{"role": "user", "content": "Say 'Sarvam AI API key is valid'"}],
        max_tokens=20
    )
    
    print("✓ SUCCESS: Sarvam AI API key is valid!")
    print(f"Response: {response.choices[0].message.content}")
    
except AuthenticationError as e:
    print("❌ AUTHENTICATION ERROR: Invalid Sarvam AI API key")
    print(f"Error details: {e}")
    print("\nPlease update your API key in the .env file")
    print("Get a new key from: https://console.sarvam.ai/")
    
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}")
    print(f"Details: {e}")

