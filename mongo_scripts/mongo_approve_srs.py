import pymongo
import os
import fitz
from langdetect import detect


MONGO_URI = os.getenv('MONGO_URI')
client = pymongo.MongoClient(MONGO_URI)
db = client['db_SDG']

def approve_sr(sr_id):
    
    # create path to document
    sr_path = '/Users/charlott/Nextcloud/Documents/SRs/' + str(sr_id)

    text = ""

    try: 
        with fitz.open(sr_path) as doc:
            
            amount_pages = len(doc)

            if amount_pages == 0:
                return False, None, None, None
            
            if amount_pages < 10:
                    text = doc.load_page(round(amount_pages/2)-1).get_text()
                    min_words = 30
            else:
                # get pages 5 and 8 to check the extratibility and language
                for page in doc.pages(5, 10, 3):
                    text = text + " " + page.get_text()
                min_words = 60

        text_length = len(text.split())

        if text_length > min_words:
            is_extractable  = True
            text_language = detect(text)

        else:
            is_extractable  = False
            text_language = None
    
    except Exception as e:
        print(e)
        return False, None, None, None

    return True, amount_pages, text_language, is_extractable





sustainability_reports = db.sustainability_reports.find({"download_successful": {"$exists": False}}).limit(7000)

i = 0
for sr in sustainability_reports:
    sr_id = sr['_id']

    download_successful, amount_pages, language, is_extractable = approve_sr(sr_id)

    sr = db.sustainability_reports.update_one(
        {"_id": sr_id},
        {"$set": {
            "download_successful": download_successful,
            "amount_pages": amount_pages,
            "language": language,
            "is_extractable": is_extractable
        }})
    
    i += 1
    print(i)


client.close()