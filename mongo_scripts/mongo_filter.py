import pymongo
import shutil
import os
import pandas as pd

from data_preperation.text.utils.encode_sdgs import encode_sdgs_multi_label
import datetime

MONGO_URI = os.getenv('MONGO_URI')
client = pymongo.MongoClient(MONGO_URI)
db = client['db_SDG']


agg = db.sustainability_reports_test.aggregate([
    {"$unwind": "$page_data"},
    {"$project": 
        {"page_number": "$page_data.page_number",
        "page_text": "$page_data.page_text",
        "page_labels": "$page_data.page_labels",
        "count_positives": {'$add':[
            "$page_data.page_labels.1", 
            "$page_data.page_labels.2",
            "$page_data.page_labels.3",
            "$page_data.page_labels.4",
            "$page_data.page_labels.5",
            "$page_data.page_labels.6",
            "$page_data.page_labels.7",
            "$page_data.page_labels.8",
            "$page_data.page_labels.9",
            "$page_data.page_labels.10",
            "$page_data.page_labels.11",
            "$page_data.page_labels.12",
            "$page_data.page_labels.13",
            "$page_data.page_labels.14",
            "$page_data.page_labels.15",
            "$page_data.page_labels.16",
            "$page_data.page_labels.17",
            ]}}
    },
    
])

print(agg)

agg = db.sustainability_reports_test.aggregate([
    {$unwind: "$page_data"},
    {$project: 
        {page_number: "$page_data.page_number",
        page_text: "$page_data.page_text",
        page_labels: "$page_data.page_labels",
        count_positives: {$add:[
            "$page_data.page_labels.1", 
            "$page_data.page_labels.2",
            "$page_data.page_labels.3",
            "$page_data.page_labels.4",
            "$page_data.page_labels.5",
            "$page_data.page_labels.6",
            "$page_data.page_labels.7",
            "$page_data.page_labels.8",
            "$page_data.page_labels.9",
            "$page_data.page_labels.10",
            "$page_data.page_labels.11",
            "$page_data.page_labels.12",
            "$page_data.page_labels.13",
            "$page_data.page_labels.14",
            "$page_data.page_labels.15",
            "$page_data.page_labels.16",
            "$page_data.page_labels.17",
            ]}}
    },
    
])

# filter actual labels -> delet  count >=12 , filter text between 100 and 520




