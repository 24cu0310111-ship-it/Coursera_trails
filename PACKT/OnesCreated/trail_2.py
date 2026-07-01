
import pandas as pd

url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(url)

'''This code downloads the Titanic dataset from the given URL and stores it in a pandas DataFrame called "titanic_df". 

Now, let's get the column names of the dataset. We can use the "columns" attribute of the DataFrame to do this. Here's the code:'''

column_names = titanic_df.columns
print(column_names)
