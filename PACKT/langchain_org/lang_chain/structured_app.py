"""
Structured AI Application with LangChain
Demonstrates: Building a complete AI application with modular components
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

class AIAssistant:
    """A structured AI assistant with multiple capabilities"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the sentiment of the following text and respond with "
            "either 'positive', 'negative', or 'neutral':\n\n{text}\n\n"
            "Sentiment:"
        )
        chain = prompt | self.llm
        response = chain.invoke({"text": text})
        return response.content.strip().lower()
    
    def summarize_text(self, text, max_words=50):
        """Summarize text to specified word count"""
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following text in {max_words} words or less:\n\n{text}\n\n"
            "Summary:"
        )
        chain = prompt | self.llm
        response = chain.invoke({"text": text, "max_words": max_words})
        return response.content.strip()
    
    def extract_entities(self, text):
        """Extract structured information from text"""
        response_schemas = [
            ResponseSchema(name="names", description="Names of people mentioned"),
            ResponseSchema(name="locations", description="Locations mentioned"),
            ResponseSchema(name="organizations", description="Organizations mentioned")
        ]
        
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_template(
            "Extract entities from the following text.\n\n"
            "{format_instructions}\n\n"
            "Text: {text}\n"
        )
        
        chain = prompt | self.llm
        response = chain.invoke({
            "text": text,
            "format_instructions": format_instructions
        })
        
        return parser.parse(response.content)
    
    def generate_response(self, user_input, context=""):
        """Generate contextual response"""
        prompt = ChatPromptTemplate.from_template(
            "Context: {context}\n\n"
            "User: {user_input}\n\n"
            "Respond helpfully and concisely."
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "user_input": user_input})
        return response.content


# Demonstration
if __name__ == "__main__":
    print("="*60)
    print("Structured AI Application Demo")
    print("="*60 + "\n")
    
    assistant = AIAssistant()
    
    # Test 1: Sentiment Analysis
    print("1. Sentiment Analysis")
    print("-" * 60)
    text1 = "I absolutely love using LangChain! It makes AI development so easy."
    sentiment = assistant.analyze_sentiment(text1)
    print(f"Text: {text1}")
    print(f"Sentiment: {sentiment}\n")
    
    # Test 2: Text Summarization
    print("2. Text Summarization")
    print("-" * 60)
    text2 = ("LangChain is a framework designed to simplify the creation of applications "
             "using large language models. It provides tools for prompt management, "
             "chains for combining multiple operations, and integrations with various "
             "AI services and data sources.")
    summary = assistant.summarize_text(text2, max_words=20)
    print(f"Original: {text2}")
    print(f"Summary: {summary}\n")
    
    # Test 3: Entity Extraction
    print("3. Entity Extraction")
    print("-" * 60)
    text3 = "John Smith works at OpenAI in San Francisco and collaborates with Google in Mountain View."
    entities = assistant.extract_entities(text3)
    print(f"Text: {text3}")
    print(f"Extracted entities: {entities}\n")
    
    # Test 4: Contextual Response
    print("4. Contextual Response Generation")
    print("-" * 60)
    context = "The user is learning about LangChain and OpenAI API integration."
    user_input = "What should I learn next?"
    response = assistant.generate_response(user_input, context)
    print(f"Context: {context}")
    print(f"User: {user_input}")
    print(f"Assistant: {response}\n")
    
    print("="*60)
    print("This demonstrates a structured approach to building AI apps:")
    print("1. Modular design with separate methods for different tasks")
    print("2. Reusable components (prompts, chains, parsers)")
    print("3. Clear separation of concerns")
    print("4. Easy to test and extend")
    print("="*60)
