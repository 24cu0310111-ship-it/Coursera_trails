"""
Main example script demonstrating image understanding with GPT-4o via OpenAI.
Shows how to:
1. Query the model with an image URL
"""

import base64
import mimetypes
import os
from dotenv import load_dotenv
from pathlib import Path

import backoff
from openai import OpenAI, RateLimitError

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@backoff.on_exception(backoff.expo, RateLimitError)
def query_model_single_turn(messages, model="gpt-4o-mini", temperature=0, **kwargs):
    """
    Query the OpenAI ChatCompletion API with exponential backoff.

    Args:
        messages (list): A list of chat messages.
        model (str): The model to use.
        temperature (float): The temperature to use.
        **kwargs: Additional keyword arguments to be passed to client.chat.completions.create().

    Returns:
        An OpenAI ChatCompletion object.
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


def query_image_from_url(image_url, question="What's happening here?", model="gpt-4o"):
    """
    Query the model with a question about an image from a URL.

    Args:
        image_url (str): The URL of the image.
        question (str): The question to ask about the image.
        model (str): The model to use.

    Returns:
        str: The model's response.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        }
    ]

    response = query_model_single_turn(messages, model=model)
    return response.choices[0].message.content


def example_image_url():
    """Example: Query model with an image URL"""
    print("=" * 60)
    print("Example 1: Image from URL")
    print("=" * 60)
    
    image_url = "https://indianatravelservices.com/blog/great-indian-festivals-celebrate/"
    question = "What's happening in this image?"
    
    response = query_image_from_url(image_url, question, model="gpt-4o")
    print(f"\nQuestion: {question}")
    print(f"Response: {response}\n")


if __name__ == "__main__":
    # Run examples
    try:
        example_image_url()
    except Exception as e:
        print(f"Error in example_image_url: {e}")
