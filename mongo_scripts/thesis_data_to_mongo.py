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

company_info_table_path = "/Users/charlott/Documents/github_repos/sdg_classification/data/un_global_compact_data_500_1500.csv"
company_info = pd.read_csv(company_info_table_path)
company_info = company_info.rename(columns={"document_file_name": "_id"})[["company_country", "company_name", "company_website", "_id"]]
company_info["_id"] = company_info["_id"].apply(lambda x: str(x))

table_path = "/Users/charlott/Documents/github_repos/sdg_classification/data/sr_data2_important.csv"
df = pd.read_csv(table_path) # ,dtype={'_id':'string'}
df = df.rename(columns={"file_name": "_id", "page": "page_number", "text": "page_text", "predictions": "sdgs"})[["_id", "page_number", "page_text", "sdgs"]]
df["_id"] = df["_id"].apply(lambda x: str(x))


# merge tables
df = df.merge(company_info, on="_id", how='left')

df.to_csv("checkcheck.csv")

df = encode_sdgs_multi_label(df, sdg_column='sdgs')
df['page_labels'] = df[[str(x) for x in range(1, 18)]].apply(lambda x: x.to_dict(), axis=1)


for sr_id in df['_id'].unique():

    
    sr_pages = df[(df['_id'] == sr_id) & (df['page_number'].notna())]

    # align columns
    sr_pages['page_number'] = sr_pages['page_number'].apply(lambda x: int(str(x)))
    labels_json = sr_pages[['page_number', 'page_text', 'page_labels']].to_dict(orient='records')


    # db.sustainability_reports_test.update_one(
    # {"_id": int(sr_id),
    #  "page_data": {"$exists": False}
    # },
    # {"$set": {
    #     "company_country": sr_pages["company_country"].iloc[0], 
    #     "company_name": sr_pages["company_name"].iloc[0], 
    #     "company_website": sr_pages["company_website"].iloc[0],
    #     "page_data": labels_json,
    #     "page_labelling": {
    #         "at": datetime.datetime(2022,10,1),
    #         "how": "correction"
    #     }
    # }}, upsert=True)


    result = db.sustainability_reports.find_one({"_id": int(sr_id)})
    if result:
        db.sustainability_reports.update_one(
        {"_id": int(sr_id),
        "page_data": {"$exists": False}
        },
        {"$set": {
            "page_data": labels_json,
            "page_labelling": {
                "at": datetime.datetime(2022,10,1),
                "how": "correction"
            }
        }})
    else:
        db.sustainability_reports.update_one(
        {"_id": int(sr_id),
        "page_data": {"$exists": False}
        },
        {"$set": {
            "company_country": sr_pages["company_country"].iloc[0], 
            "company_name": sr_pages["company_name"].iloc[0], 
            "company_website": sr_pages["company_website"].iloc[0],
            "page_data": labels_json,
            "page_labelling": {
                "at": datetime.datetime(2022,10,1),
                "how": "correction"
            }
        }}, upsert=True)      