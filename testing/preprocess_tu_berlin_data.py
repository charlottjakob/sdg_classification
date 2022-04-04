import pandas as pd
import regex as re

# load data
df = pd.read_csv('files/tu_berlin_data.csv')

# delete empty cells
df = df[df['sdgs'].notna()]

# get digits from sdg columns
df['sdgs'] = df['sdgs'].apply(lambda x: re.findall('\-\s(\d{1,2})', x))

# save df
df.to_csv('files/tu_berlin_data.csv')


print('end')