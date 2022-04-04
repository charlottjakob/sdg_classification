import scrapy
import pandas as pd


class COPSpider(scrapy.Spider):
    """Spider to scrapy companies and ther commited sdgs from ungloablcompact."""

    name = 'cop_spider'
    custom_settings = {
        "ITEM_PIPELINES": {
            "scraping.scraping.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "files/cop_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):
        """.

        Yields:
            request
        """

        un_global_compact_data = pd.read_csv("files/un_global_compact_data.csv")

        for idx, company in un_global_compact_data.iterrows():

            yield scrapy.Request(
                url=company['cop_link'],
                callback=self.parse,
                cb_kwargs={
                        "company_name": company['company_name'],
                        "company_website": company['company_website'],
                        "company_country": company['company_country'],
                        "sdgs": company['sdgs'],
                },
            )


    def parse(self, response, **kwargs):

        print('hello')