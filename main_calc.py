# basic packages
import pandas as pd
from pathlib import Path
import regex as re
# import tensorflow as tf 
# import tensorflow_hub as hub
# import tensorflow_text as text

# local functions
from training_data.build_basic_dict import combine_un_data
from training_data.clean_dict import clean_basic
from training_data.vector_embedding import word2vec
# from training_data.extract_keywords import extract_keywords
from utils.tokenize import tokenize_to_sentences
from test_data.extract_examples import company_to_website_sentences


print('start')

df = combine_un_data()
df = clean_basic(df)
# df = extract_keywords(df)
df = tokenize_to_sentences(df)


df = tokenize_to_sentences(df)


# down-sampling
count_sentences_min = min(df.groupby('goal')['goal'].count())
df= df.groupby('goal').sample(n=count_sentences_min) 

df.to_csv('data/training_data.csv')

company_to_website_sentences("Beauty Industry Group")

print('end')









