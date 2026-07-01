from openai import OpenAI
import os

#Step 1: Write the prompt template
prompt_template_v4 = """
Answer the python question in the triple backticks. 

```
{0}
```

Answer the python question in the triple backticks. 

Use the following template. 

Step One: 
 - What is the question really asking?
 - Is it really a Python question?
 - What do you know about the answer to this? 
 - Is the answer to the question part of your knowledge?
Step Two:
 - Answer the question. Or if you don't know the answer, reply with "I don't know and I don't want your hallucinogens!" if you don't know the answer.
     Take a deep breath and explain step by step.
     This is very important to my career.
     Embrace challenges as opportunities for growth. Each obstacle that you overcome brings you closer to success.
"""
 #Step 2: Insert the question into the template
our_question = "How do I fit a random forest model in Python?"
our_prompt = prompt_template_v4.format(our_question)
print(our_prompt)

#Step 3: Make the API call

#3.0 Put our prompt into a dictionary object. 
#      We will discuss this structure in a future lesson.
messages = [{"role": "user", "content": our_prompt}]

#3.1 Query the API
client = OpenAI(api_key=os.getenv("pmpt_69835602d7cc8196a949bb95fa19f4600c146bdc13a3f841"))
response = client.chat.completions.create(
        model="gpt-4.1-mini",    
        messages=messages,
        temperature=0) 
#Step 4: Print the response
print(response.choices[0].message.content)