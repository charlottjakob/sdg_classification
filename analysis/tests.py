import pandas as pd
import os

print(os.getcwd())
df = pd.read_csv('files/un_data.csv')

print(df)
