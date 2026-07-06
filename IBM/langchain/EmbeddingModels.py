"""Why we use Embeddings in AI
Computers cannot natively read or compare the semantic meaning of words. 
Embedding models convert strings of words into high-dimensional geometric vectors. Words with similar meanings wind up closer together in this mathematical space.

2. Why TRUNCATE_INPUT_TOKENS: 3 is used here
Embedding models have a strict ceiling on the maximum length of text they can process at one time (often 512 or 8192 tokens). 
Setting this value to 3 is highly unusual for real production code, but it is required in your lab to demonstrate truncation behavior. 
It allows you to see exactly how the model drops extra words when an input exceeds the allocated limit without needing to type out a massive paragraph.
  
3. Why RETURN_OPTIONS: {"input_text": True} is required
By default, an embedding API just sends back a massive, anonymous list of numbers (e.g., [0.012, -0.234, 0.891, ...]). 
If you send a batch of 50 sentences at once, it is very easy to lose track of which array of numbers belongs to which sentence. 
Turning this option on binds your original string data directly to the numeric output matrix, making debugging and data mapping much easier.
  In simple :
   The Solution (With RETURN_OPTIONS: {"input_text": True})When you turn this setting on, you are telling the AI: 
   "When you give me the numbers, glue my original text right next to them."Instead of just getting an anonymous wall of math,
     your output is explicitly mapped out like this:"I love machine learning" $\rightarrow$ [0.012, -0.234, 0.891, ...]"The weather is nice" $\rightarrow$ [0.456, 0.112, -0.987, ...]"""
# Import the EmbedTextParamsMetaNames class from ibm_watsonx_ai.metanames module
# This class provides constants for configuring Watson embedding parameters
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# Configure embedding parameters using a dictionary:
# - TRUNCATE_INPUT_TOKENS: Limit the input to 3 tokens (very short, possibly for testing)
# - RETURN_OPTIONS: Request that the original input text be returned along with embeddings
embed_params = {
 EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
 EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

# Import the WatsonxEmbeddings class from langchain_ibm module
# This provides an integration between LangChain and IBM's Watson AI services
from langchain_ibm import WatsonxEmbeddings

# Create a WatsonxEmbeddings instance with the following configuration:
# - model_id: Specifies the "slate-125m-english-rtrvr-v2" embedding model from IBM
# - url: The endpoint URL for the Watson service in the US South region
# - project_id: The Watson project ID to use ("skills-network")
# - params: The embedding parameters configured earlier
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

texts = [text.page_content for text in chunks]

embedding_result = watsonx_embedding.embed_documents(texts)
embedding_result[0][:5]