import pymongo
import shutil
import os
import pandas as pd
import fitz
from data_preperation.text.utils.encode_sdgs import encode_sdgs_multi_label
import datetime

MONGO_URI = os.getenv('MONGO_URI')
client = pymongo.MongoClient(MONGO_URI)
db = client['db_SDG']

# get 10 valid reports from database
package_number = 18

table_path = '/Users/charlott/Nextcloud/Documents/Labeling/Package_' + str(package_number)  + "_done/labelling_table.csv"

# first delete the rows that are empty
df = pd.read_csv(table_path,dtype={'_id':'string'})

df = df[~(df['sdgs'].isna() & df['_id'].isna())]
df['_id'].fillna(method="ffill",inplace=True)
df['page_number'] = df['page_number'].astype('Int64')


df = encode_sdgs_multi_label(df, sdg_column='sdgs')
df['page_labels'] = df[[str(x) for x in range(1, 18)]].apply(lambda x: x.to_dict(), axis=1)


for sr_id in df['_id'].unique():

    sr_path = '/Users/charlott/Nextcloud/Documents/SRs/' + str(sr_id)
    with fitz.open(sr_path) as doc:
            

        sr_pages = df[(df['_id'] == sr_id) & (df['page_number'].notna())]
        sr_pages['page_text'] = ""
        for i, page in sr_pages.iterrows():
            sr_pages.at[i,'page_text'] = doc.load_page(page['page_number']-1).get_text()
    
    sr_pages['page_number'] = sr_pages['page_number'].apply(lambda x: int(str(x)))
    

    labels_json = sr_pages[['page_number', 'page_text', 'page_labels']].to_dict(orient='records')


    db.sustainability_reports.update_one(
    {"_id": int(sr_id)},
    {"$set": {
        "page_data": labels_json,
        "page_labelling": {
            "at": datetime.datetime.now(),
            "how": "manual"
        }
    }}, upsert=True)
