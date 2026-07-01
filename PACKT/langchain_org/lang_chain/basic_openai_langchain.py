"""
Basic OpenAI API & LangChain Integration
Demonstrates: Simple chat completion using LangChain with OpenAI
"""

import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


def create_llm() -> ChatOpenAI:
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        api_key=api_key,
    )


def main() -> None:
    llm = create_llm()

    # Example 1: Simple chat completion
    print("Example 1: Simple Chat Completion")
    print("-" * 50)
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Explain what LangChain is in one sentence."),
    ]
    response = llm.invoke(messages)
    print(f"Response: {response.content}\n")

    # Example 2: Question answering
    print("Example 2: Question Answering")
    print("-" * 50)
    question = "What is the capital of France?"
    response = llm.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {response.content}\n")

    # Example 3: Using templates
    print("Example 3: Using Prompt Templates")
    print("-" * 50)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in {subject}."),
            ("human", "{question}"),
        ]
    )

    formatted_prompt = template.format_messages(
        subject="Python programming",
        question="What are the benefits of using virtual environments?",
    )

    response = llm.invoke(formatted_prompt)
    print(f"Answer: {response.content}\n")


if __name__ == "__main__":
    main()
