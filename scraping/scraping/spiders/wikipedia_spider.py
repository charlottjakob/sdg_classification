import scrapy
from scraping.scraping.items import WikipediaItem
from scrapy.loader import ItemLoader
import re
import pandas as pd

class WikipediaSpider(scrapy.Spider):
    name = 'wikipedia_spider'
    allowed_domains = ['wikipedia.org']
    custom_settings = {
        "ITEM_PIPELINES": {
            "scraping.scraping.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "data/wikipedia_data.csv": {"format": "csv"},
        }
    }

    def start_requests(self):
        """Request all pages one by one to extract the links of companies.

        Yields:
            request
        """
        # iterate thourgh the pages form last to first
        terms = ['capitalism', 'finance', 'engineering', 'marketing', 'business_development', 'strategic_management']
        # pages = [1629]
        for term in terms:

            # create url and trigger request
            url = 'https://en.wikipedia.org/wiki/' + str(term)
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                cb_kwargs={'term': term},
            )

    def parse(self, response, **kwargs):

        # Initialize loader
        loader = ItemLoader(item=WikipediaItem(), selector=response)

        paragraph_paths = response.xpath('//p')
        paragraphs = []
        for paragraph_path in paragraph_paths:
            text_parts = paragraph_path.xpath('.//text()').extract()
            text = ''.join(text_parts)
            paragraphs.append(text)

        # add values and website content to loader
        loader.add_value("text", paragraphs)
        loader.add_value("term", kwargs["term"])

        # trigger pipeline
        yield loader.load_item()
