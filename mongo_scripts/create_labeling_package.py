import pymongo
import shutil
import os
import pandas as pd

mongo_uri = os.getenv('MONGO_URI')
client = pymongo.MongoClient(mongo_uri)
db = client['db_SDG']

# get 10 valid reports from database
package_number = 19

# important_company_names = [
#     "Deutsche Bahn AG",
#     "TotalEnergies",
#     "Vattenfall AB",
#     "The Coca-Cola Company",
#     "Bonava AB",
#     "Microsoft Corporation",
#     "NVIDIA Corporation",
#     "Volkswagen AG",
#     "Siemens AG",
#     "Allianz SE",
#     "Daimler Truck AG",
#     "Deutsche Telekom AG",
#     "BASF SE",
#     "Merck KGaA",
#     "Deutsche Post DHL Group",
#     "BMW AG",
# ]

# pages between 10 and 250, english, extractable, 
sustainability_reports = db.sustainability_reports.find({
    "is_extractable": True,
    "language": "en",
    "amount_pages": {"$gt":10, "$lt": 250},
    "page_data": {"$exists": False},
    # "company_name": {"$in": important_company_names}
},
{"company_name":1, "_id":1, "language":1,"amount_pages":1, "is_extractable":1, "amount_pages":1, "page_data":1 }
) # .limit(50)
sr_list = list(sustainability_reports)

# create directory
package_path = '/Users/charlott/Nextcloud/Documents/Labeling/Package_' + str(package_number)
if not os.path.exists(package_path):
    os.makedirs(package_path)



# add company name and report number to dataframe and save as table
df = pd.DataFrame(sr_list).sort_values('_id')

df['page_number'] = ''
df['sdgs'] = ''
df['comment'] = ''

df.to_csv(package_path + "/labelling_table.csv")


# copy the reports with .pdf to empty labeling-folder package
for sr in sr_list:
    
    sr_path = '/Users/charlott/Nextcloud/Documents/SRs/' + str(sr['_id'])
    sr_path_copy = package_path + "/" + str(sr['_id']) + ".pdf"

    shutil.copy(sr_path, sr_path_copy)



# in another script:
# read table -> extract text and update mongodb






