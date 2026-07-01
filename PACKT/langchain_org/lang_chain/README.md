# OpenAI API & LangChain Framework - Deep Dive

This directory contains simple examples demonstrating the key concepts of LangChain framework with OpenAI API integration.

## Files Overview

### 1. `basic_openai_langchain.py`
**Concept**: Basic OpenAI API integration with LangChain
- Simple chat completions
- Question answering
- Using prompt templates

### 2. `document_query.py`
**Concept**: Query documents using RAG (Retrieval Augmented Generation)
- Document splitting
- Creating embeddings
- Vector store setup
- Retrieval-based question answering

### 3. `chatbot_test.py`
**Concept**: Testing chatbot interactions
- Conversation without memory
- Conversation with memory
- Custom chatbot roles
- Viewing conversation history

### 4. `structured_app.py`
**Concept**: Structuring AI-powered applications
- Modular application design
- Sentiment analysis
- Text summarization
- Entity extraction
- Contextual responses

### 5. `chat_your_data.py` (Already exists)
**Concept**: Complete conversational RAG system
- Load multiple document types (.txt, .md, .csv)
- Interactive chat loop with history
- Source document tracking

## Prerequisites

Install required packages:
```bash
pip install langchain langchain-openai langchain-community python-dotenv chromadb
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Examples

```bash
# Basic integration
python basic_openai_langchain.py

# Document querying
python document_query.py

# Chatbot testing
python chatbot_test.py

# Structured application
python structured_app.py

# Chat with your documents
python chat_your_data.py
```

## Key Concepts Covered

1. **LangChain Framework**: A framework for building LLM-powered applications
2. **OpenAI Integration**: Using OpenAI's API through LangChain
3. **Document Querying**: RAG pattern for question-answering over documents
4. **Chatbot Interactions**: Building conversational AI with memory
5. **Structured Applications**: Modular, maintainable AI application design
