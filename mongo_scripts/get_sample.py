# get all pages with exact 2 positive labels
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

MONGO_URI = os.getenv('MONGO_URI')
client = pymongo.MongoClient(MONGO_URI)
db = client['db_SDG']

# get 10 valid reports from database
package_number = 17


# pages between 10 and 250, english, extractable, 
sustainability_reports = db.sustainability_reports.find({
    "is_extractable": True,
    "language": "en",
    "amount_pages": {"$gt":10, "$lt": 250},
    "page_data": { "$exists": True, "$ne": [] }
},
{"page_data":1, "_id":1}
) # .limit(50)
sr_list = list(sustainability_reports)



report_pages = [page for sr in sr_list for page in sr['page_data'] ]

df = pd.DataFrame(report_pages)
labels = df['page_labels'].apply(pd.Series).astype('Int64')
count = labels.sum(axis=1).rename('count_positive')
df = pd.concat([df.drop(['page_labels'], axis=1), labels, count], axis=1)

df = df[df['count_positive'] == 2.0]


# count combinations and choose the one mostly present
sdgs = [str(sdg) for sdg in range(1,18)]
for sdg in sdgs:
    df[sdg] = np.where(df[sdg] == 1, sdg + ",", "")

df['labels_str'] = df[sdgs].sum(axis=1)
print(df['labels_str'].value_counts())


# filter text length < 500 words
df['word_count'] = df['page_text'].apply(lambda x: len(x.split(" ")))
df = df[df['word_count'] <= 500]

# 7,13

for i, page in df[df['labels_str'] == "7,13,"].iterrows():
    print("-"*30)
    print(page['page_text'])