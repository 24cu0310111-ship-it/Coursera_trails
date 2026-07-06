from langchain_core.runnables import RunnablePassthrough

// Create each individual chain with the pipe operator

location_chain_lcel = (

PromptTemplate.from_template(location_template)

| mixtral_llm

| StrOutputParser()

)


dish_chain_lcel = (

PromptTemplate.from_template(dish_template)

| mixtral_llm

| StrOutputParser()

)


time_chain_lcel = (

PromptTemplate.from_template(time_template)

| mixtral_llm

| StrOutputParser()

)


overall_chain_lcel = (

RunnablePassthrough.assign(meal=lambda x: location_chain_lcel.invoke({“location”: x[“location”]}))

| RunnablePassthrough.assign(recipe=lambda x: dish_chain_lcel.invoke({“meal”: x[“meal”]}))

| RunnablePassthrough.assign(time=lambda x: time_chain_lcel.invoke({“recipe”: x[“recipe”]}))

)


// Run the chain

result = overall_chain_lcel.invoke({“location”: “China”})

pprint(result)

