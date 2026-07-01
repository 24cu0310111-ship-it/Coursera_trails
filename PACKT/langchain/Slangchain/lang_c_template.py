
import os 
import asyncio
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# Load .env file - automatically finds it in parent directories
env_file = find_dotenv()
if env_file:
    print(f"📁 Loading .env from: {env_file}")
    load_dotenv(env_file)
else:
    print("⚠️  No .env file found, trying current directory...")
    load_dotenv()

# Verify API key is loaded
api_key = os.getenv("SarvamAI_API_KEY")
if not api_key:
    print("\n❌ ERROR: SarvamAI_API_KEY not found!")
    print("   Make sure your .env file contains: SarvamAI_API_KEY=your_key_here")
    exit(1)

# Use LangChain with Sarvam AI (OpenAI-compatible endpoint)
llm = ChatOpenAI(
    api_key=api_key,  # Pass API key as string
    base_url="https://api.sarvam.ai/v1",  # Sarvam AI endpoint
    model="sarvam-m",  # Options: "sarvam-m", "sarvam-30b", "sarvam-105b"
    temperature=0.6,
    max_tokens=200
)

# Available engineering groups
ENGINEERING_GROUPS = {
    "1": "Software Engineer",
    "2": "DevOps Engineer",
    "3": "Data Engineer",
    "4": "Machine Learning Engineer",
    "5": "Frontend Engineer",
    "6": "Backend Engineer",
    "7": "Full Stack Engineer",
    "8": "Cloud Engineer",
    "9": "Cybersecurity Engineer",
    "10": "embedded systems Engineer"
}

def display_options():
    """Display available engineering groups"""
    print("\n=== Engineering Groups ===")
    for key, value in ENGINEERING_GROUPS.items():
        print(f"{key}. {value}")
    print()

def get_user_input():
    """Get engineering group from user"""
    display_options()
    while True:
        choice = input("Select your engineering group (1-10): ").strip()
        if choice in ENGINEERING_GROUPS:
            return ENGINEERING_GROUPS[choice]
        else:
            print("Invalid choice. Please select a number between 1-10.")

async def generate_response(engineering_group):
    """Generate customized response based on engineering group"""
    prompt = f"Give me 3 best Twitter Handle names for a {engineering_group} that are creative, professional, and catchy. Format as a numbered list."
    
    print(f"\n📧 Generating Twitter handles for: {engineering_group}")
    print("-" * 50)
    
    response = await llm.ainvoke(prompt)
    return response.content

async def main():
    """Main function"""
    print("🚀 Twitter Handle Generator for Engineers")
    print("=" * 50)
    
    engineering_group = get_user_input()
    result = await generate_response(engineering_group)
    print(result)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
