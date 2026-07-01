"""
Test script to verify HUGGINGFACE_API_KEY is working
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

def test_huggingface_api_key():
    """Test if the HUGGINGFACE_API_KEY is valid and working"""
    
    # Get the API key
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        print("❌ ERROR: HUGGINGFACE_API_KEY not found in .env file")
        return False
    
    print(f"✓ API Key found (first 10 chars): {api_key[:10]}...")
    
    try:
        # Initialize HuggingFace client
        client = InferenceClient(token=api_key)
        
        # Make a simple API call
        print("\n🔄 Testing API connection with a simple message...")
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "user", "content": "Say 'API is working!' in one sentence."}
            ],
            max_tokens=50
        )
        
        # Check if we got a valid response
        if response.choices and response.choices[0].message.content:
            print(f"✓ API Response: {response.choices[0].message.content}")
            print("\n✅ SUCCESS: HUGGINGFACE_API_KEY is working correctly!")
            return True
        else:
            print("❌ ERROR: No valid response from API")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        print("\n💡 TIP: Make sure you have a valid HuggingFace token from https://huggingface.co/settings/tokens")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("HuggingFace API Key Test")
    print("=" * 50)
    test_huggingface_api_key()
