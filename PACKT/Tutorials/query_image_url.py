"""
Query model with image URLs using OpenRouter.
Handles API calls with images referenced by URL.
"""

from openai import RateLimitError
from config import openrouter
import backoff


@backoff.on_exception(backoff.expo, RateLimitError)
def query_model_single_turn(prompt, model="gpt-4o-mini", temperature=0, **kwargs):
    """
    This function queries the OpenRouter ChatCompletion API, with exponential backoff.
    
    Args:
        prompt (str or list): The prompt or a list of messages.
        model (str): The type of model to use. The default is "gpt-4o-mini".
        temperature (float): The temperature to use. The default is 0.
        **kwargs: Additional keyword arguments to be passed to openrouter.chat.completions.create()
    
    Returns:
        An OpenRouter ChatCompletion object.
    """
    # Set up the messages list
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    
    return openrouter.chat.completions.create(
        model=model,    
        messages=messages,
        temperature=temperature,
        **kwargs
    )


def query_image_from_url(image_url, question="What's happening here?", model="gpt-4o"):
    """
    Query the model with a question about an image from a URL.
    
    Args:
        image_url (str): The URL of the image
        question (str): The question to ask about the image
        model (str): The model to use
    
    Returns:
        str: The model's response
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
