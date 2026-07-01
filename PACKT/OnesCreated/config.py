"""
Configuration and setup for Image Understanding API calls.
Handles API key setup and common imports.
"""

import openai
import os
import backoff

# Configure OpenAI API key from environment variable
openai.api_key = os.getenv('MY_API_KEY')
