"""
Query model with uploaded images using OpenRouter.
Handles API calls with images uploaded from disk.
"""

import base64
import requests
import backoff


@backoff.on_exception(backoff.expo, KeyError)
def upload_image_and_query_model_single_turn(prompt, path_to_image, api_key, model="gpt-4o-mini", temperature=0):
    """
    This function queries the model with a text prompt and an image. 
    The image is uploaded from disk using OpenRouter API.
    
    Args:
        prompt (str): The prompt
        path_to_image (str): The path to the image
        api_key (str): The OpenRouter API key
        model (str): The type of model to use. The default is "gpt-4o-mini".
        temperature (float): The temperature to use. The default is 0.
    
    Returns:
        str: The string of the model's reply.
    """
    # 1. Read and encode the image
    with open(path_to_image, "rb") as image_file:
        the_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # 2. Set up the messages list (only one message)
    messages_list = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{0}".format(the_encoded_image)
                    }
                }
            ]
        }
    ]
    
    # 3. Prepare the payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {0}".format(api_key)
    }
    payload = {
        "model": model,
        "messages": messages_list,
        "temperature": temperature
    }
    
    # 4. Make the API call to OpenRouter
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    
    # 5. Parse the output
    response_dict = response.json()
    
    # 6. Return the text of the reply
    # If we have any problems here, they will be caught by the @backoff decorator
    response_text = response_dict["choices"][0]["message"]["content"] 
    return response_text
