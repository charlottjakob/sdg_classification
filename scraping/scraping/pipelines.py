# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.pipelines.files import FilesPipeline
import os
import pymongo
from bson.objectid import ObjectId


class BasicPipeline:
    def process_item(self, item, spider):
        return item


class DownloadPdfPipeline(FilesPipeline):
 
    def file_path(self, request, response=None, info=None):
        file_name: str = request.url.split("?")[-1]
        return file_name

    pass


class MongoPipeline:

    collection_name = 'sustainability_reports'

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DB')
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        
        sr_dict = ItemAdapter(item).asdict()
        sr_dict["_id"] = int(sr_dict['document_file_name'])
        
        sr_dict.pop('file_urls', None)
        sr_dict.pop('document_file_name', None)  

        self.db[self.collection_name].insert_one(sr_dict)

        return item