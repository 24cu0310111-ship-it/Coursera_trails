from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ibm import WatsonxLLM

# Sample product reviews for testing
positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing.
The built-in grinder saves me so much time in the morning, and the programmable timer means
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use,
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours.
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

# Step 1: Define the prompt templates for each processing step
sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

response_template = """Write a helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""

# Create prompt templates for each step
template_sqlc_1 = PromptTemplate(template=sentiment_template, input_variables=["review"])
template_sqlc_2 = PromptTemplate(template=summary_template, input_variables=["review", "sentiment"])
template_sqlc_3 = PromptTemplate(template=response_template, input_variables=["review", "sentiment", "summary"])

# LCEL templates
template_lcel_1 = PromptTemplate.from_template(sentiment_template)
template_lcel_2 = PromptTemplate.from_template(summary_template)
template_lcel_3 = PromptTemplate.from_template(response_template)

# IBM Watsonx LLM
llm = WatsonxLLM(
    model_id="google/flan-ul2",
    url="https://us-south.ml.cloud_ibm.com",
    project_id="skills-network",
    instance_id="openshift",
)

# PART 1: Traditional Chain Approach
sentiment_chain = LLMChain(llm=llm, prompt=template_sqlc_1, output_key="sentiment")
summary_chain = LLMChain(llm=llm, prompt=template_sqlc_2, output_key="summary")
response_chain = LLMChain(llm=llm, prompt=template_sqlc_3, output_key="response")

traditional_chain = SequentialChain(
    chains=[sentiment_chain, summary_chain, response_chain],
    input_variables=["review"],
    output_variables=["sentiment", "summary", "response"],
    verbose=True,
)

# PART 2: LCEL Approach
sentiment_chain_lcel = template_lcel_1 | llm | StrOutputParser()
summary_chain_lcel = template_lcel_2 | llm | StrOutputParser()
response_chain_lcel = template_lcel_3 | llm | StrOutputParser()

lcel_chain = (
    RunnablePassthrough.assign(
        sentiment=lambda x: sentiment_chain_lcel.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        summary=lambda x: summary_chain_lcel.invoke(
            {"review": x["review"], "sentiment": x["sentiment"]}
        )
    )
    | RunnablePassthrough.assign(
        response=lambda x: response_chain_lcel.invoke(
            {
                "review": x["review"],
                "sentiment": x["sentiment"],
                "summary": x["summary"],
            }
        )
    )
)


# Test both implementations
def test_chains(review):
    """Test both chain implementations with the given review"""
    print("\n" + "=" * 50)
    print(f"TESTING WITH REVIEW:\n{review[:100]}...\n")

    print("TRADITIONAL CHAIN RESULTS:")
    traditional_result = traditional_chain.invoke({"review": review})
    print(f"Sentiment: {traditional_result['sentiment']}")
    print(f"Summary: {traditional_result['summary']}")
    print(f"Response: {traditional_result['response']}")

    print("\nLCEL CHAIN RESULTS:")
    lcel_result = lcel_chain.invoke({"review": review})
    print(f"Sentiment: {lcel_result['sentiment']}")
    print(f"Summary: {lcel_result['summary']}")
    print(f"Response: {lcel_result['response']}")

    print("=" * 50)


# Run tests
test_chains(positive_review)
test_chains(negative_review)


"""
Expected output:
==================================================
TESTING WITH REVIEW:
I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing.
The built-in gr...

TRADITIONAL CHAIN RESULTS:
Sentiment: positive
Summary: ...
Response: ...

LCEL CHAIN RESULTS:
Sentiment: positive
Summary: ...
Response: ...
==================================================

==================================================
TESTING WITH REVIEW:
Disappointed with this laptop. It's constantly overheating after just 30 minutes of use,
and the bat...

TRADITIONAL CHAIN RESULTS:
Sentiment: negative
Summary: ...
Response: ...

LCEL CHAIN RESULTS:
Sentiment: negative
Summary: ...
Response: ...
==================================================

1. Compare the flexibility and readability of both approaches
- Traditional SequentialChain:
  - More readable for beginners because the steps are clearly arranged in order.
  - Good for simple, linear workflows.
  - Less flexible when you want to reuse or dynamically compose parts of the pipeline.

- LCEL:
  - More flexible because you can combine components easily and reuse them in different ways.
  - Cleaner and more compact for advanced workflows.
  - Slightly less beginner-friendly because the chain composition is more functional and abstract.

2. Advantages and disadvantages of each method
- SequentialChain
  - Advantages:
    - Easy to understand
    - Good for step-by-step workflows
    - Clear structure for multi-stage processing
  - Disadvantages:
    - More verbose
    - Less reusable
    - Harder to modify dynamically

- LCEL
  - Advantages:
    - Compact and modern
    - Highly flexible
    - Easier to compose and reuse
  - Disadvantages:
    - Slightly harder for beginners
    - Requires understanding of runnable composition
    - Can be less obvious than a simple sequential layout 

"""