from data_preperation.text.build_scientific_dataset import combine_training_data
from data_preperation.text.text_cleaning import base_cleaning
from data_preperation.text.utils.encode_sdgs import encode_sdgs_multi_label
from data_preperation.text.data_balancing import balance_with_ratio

# basics
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

CLASS_NAMES = [str(number) for number in np.arange(1, 18)]

# 1. Build and Clean Scientific Data and save trainingsdata_1
# build
si_train = combine_training_data(nosdg_data_included=True)

# clean
si_train = base_cleaning(si_train)

# encode labels
si_train = encode_sdgs_multi_label(si_train)

# filter for relevant columns
si_train = si_train[['data_origin', 'text', 'sdg', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']]


# 10. Clean Sustainability Report Data and save testingdata_1
# load data
sr_data = pd.read_csv("data/sr_data3.csv")

# clean
sr_data = base_cleaning(sr_data)

# encode labels
sr_data = encode_sdgs_multi_label(sr_data, sdg_column='predictions')

# delete pages with >= 12 SDGs
sr_data['count_sdgs'] = sr_data[[str(x) for x in range(0, 18, 1)]].sum(axis=1)
sr_data = sr_data[sr_data['count_sdgs'] <= 11]
print('Amount after deleting sdgs > 11: ', len(sr_data))

sr_data['text_length'] = sr_data['text'].apply(lambda x: len(str(x).split()))

# > 100 to be sure that engough information is included
sr_data = sr_data[(sr_data['text_length'] >= 100)]
print('Amount after deleting text length < 100: ', len(sr_data))

# < 520 because transformers max input length is 517
sr_data = sr_data[(sr_data['text_length'] < 520)]
print('Amount after deleting text length > 520: ', len(sr_data))

# add data origin
sr_data['data_origin'] = 'sustainability report'

# filter for relevant columns
sr_data_clean = sr_data[['data_origin', 'text', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']]

sr_data_clean['data_origin'] = 'sustainability report'
sr_data_clean.to_csv('data/sr_data_clean.csv')


# # 11. Build Trainingsdata_2 and Testingdata_2
# suffle text from sustainability reports
sr_data = sr_data.sample(frac=1)

# split into train, val and test set
X_train, y_train, X_val_test, y_val_test = iterative_train_test_split(sr_data[['text']].values, sr_data[CLASS_NAMES].values, test_size=0.4)
X_val, y_val, X_test, y_test = iterative_train_test_split(X_val_test, y_val_test, test_size=0.5)

# combine datasets
# training part of SR data
sr_train = pd.DataFrame()
sr_train['text'] = X_train.reshape(-1)
sr_train[CLASS_NAMES] = y_train
sr_train['data_origin'] = 'sustainability report'


# testing data
text_test = pd.DataFrame()
text_test['text'] = X_test.reshape(-1)
text_test[CLASS_NAMES] = y_test
text_test['data_origin'] = 'sustainability report'

# validation data
text_val = pd.DataFrame()
text_val['text'] = X_val.reshape(-1)
text_val[CLASS_NAMES] = y_val
text_test['data_origin'] = 'sustainability report'


# prepera for training combinations
sr_train['data_ml'] = 'train'
si_train['data_ml'] = 'train'
text_test['data_ml'] = 'test'
text_val['data_ml'] = 'val'

# save test dataset
text_test.to_csv('data/text_test.csv')
text_val.to_csv('data/text_val.csv')

# combine scientific data with majority of sustainability report data to get trainingdata_2
text_train_1 = si_train
text_train_2 = pd.concat([si_train, sr_train], axis=0, ignore_index=True)

# # Balancing
ratios = [0.25, 0.50, 0.75, 0.9]

# add validation data
text_train_1 = balance_with_ratio(text_train_1, ratios)
print('Balancing done: set 1')
# # text_train_val_1 = pd.concat([text_train_1, sr_val], axis=0, ignore_index=True)
text_train_1.to_csv('data/text_train_1.csv')


text_train_2 = balance_with_ratio(text_train_2, ratios)
print('Balancing done: set 2')
# # text_train_val_2 = pd.concat([text_train_2, sr_val], axis=0, ignore_index=True)
text_train_2.to_csv('data/text_train_2.csv')
