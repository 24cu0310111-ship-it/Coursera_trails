# Import the necessary packages
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Specify the model and project settings 
# (make sure the model you wish to use is commented out, and other models are commented)
#model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503' # Specify the Mixtral model
model_id = 'meta-llama/llama-3-2-11b-vision-instruct' # Specify llama model

# Set the necessary parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # Specify the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

project_id = "skills-network"
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}

# Create a ModelInference instance and invoke it directly
WatsonxModel = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id,
)

# Get the query from the user input
query = input("Please enter your query: ")

# Print the generated response
result = WatsonxModel.invoke(query)
print(result[0])


