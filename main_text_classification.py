# basic packages
import pandas as pd
from pathlib import Path
import regex as re
import json
from training.combine_source_data import combine_training_data
from training.cleaning import clean_training_data
from testing.cleaning import filter_english_language
import matplotlib.pyplot as plt
import os
import plotly.express as px
from training.utils.encode_sdgs import encode_sdgs_multi_label, encode_sdgs_multi_class
from sklearn.model_selection import train_test_split



print('start')

# load base data
df = combine_training_data(True)
df = clean_training_data(df)

# df = encode_sdgs_multi_class(df)

# save data for multi-class training
df.reset_index(inplace=True, drop=True)
df.to_csv('data/training_data_multi_class.csv')


# prepare data for multi-label
training_data = encode_sdgs_multi_label(df, 'sdg', ',')


# load report page data
df = pd.read_csv('/Users/charlottjakob/Documents/github_repos/sdg_classification/data/report_page_data.csv')
df_old = pd.read_csv('/Users/charlottjakob/Documents/github_repos/sdg_classification/data/report_page_data_old.csv')[['text','sdgs']]

# combine data 
df = pd.concat([df, df_old], ignore_index=True, axis=0)

# filter for english
df = filter_english_language(df)

# filter for pages with a certain amount of text
min_n_words = 200
df['n_words'] = df['text'].apply(lambda x: len(x.split(' ')))
df = df[df['n_words'] >= min_n_words]

# base cleaning
df = clean_training_data(df)
df = df[df['sdgs'].notna()]
df = encode_sdgs_multi_label(df, 'sdgs', ',')

df.to_csv('data/total_page_texts.csv')

pages_train , testing_data = train_test_split(df, test_size=0.1, shuffle=True)

training_data_expanded = pd.concat([training_data, pages_train], ignore_index=True, axis=0)

# save data
training_data.reset_index(inplace=True, drop=True)
training_data[['text',*[str(x) for x in range(18)]]].to_csv('data/training_data.csv')

training_data_expanded.reset_index(inplace=True, drop=True)
training_data_expanded[['text',*[str(x) for x in range(18)]]].to_csv('data/training_data_expanded.csv')

testing_data.reset_index(inplace=True, drop=True)
testing_data[['text',*[str(x) for x in range(18)]]].to_csv('data/testing_data.csv')


print('end')












