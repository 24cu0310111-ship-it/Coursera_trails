"""
Test script to verify OPENAI_API_KEY is working
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def test_openai_api_key():
    """Test if the OPENAI_API_KEY is valid and working"""
    
    # Get the API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        return False
    
    print(f"✓ API Key found (first 20 chars): {api_key[:20]}...")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make a simple API call
        print("\n🔄 Testing API connection with a simple message...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API is working!' in one sentence."}
            ],
            temperature=0.5,
            max_tokens=50
        )
        
        # Check if we got a valid response
        if response.choices and response.choices[0].message.content:
            print(f"✓ API Response: {response.choices[0].message.content}")
            print("\n✅ SUCCESS: OPENAI_API_KEY is working correctly!")
            return True
        else:
            print("❌ ERROR: No valid response from API")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_api_key()
