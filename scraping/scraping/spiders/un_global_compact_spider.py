"""Scrape companies from un global compact webpage and the sdgs they commited to."""

import scrapy
from scraping.scraping.items import UNGlobalCompactItem
from scrapy.loader import ItemLoader


class UNGlobalCompactSpider(scrapy.Spider):
    """Spider to scrapy companies and ther commited sdgs from ungloablcompact."""

    name = 'un_global_compact_spider'
    allowed_domains = ['unglobalcompact.org']
    custom_settings = {
        "ITEM_PIPELINES": {
            "scraping.scraping.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "files/un_global_compact_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):
        """Request all pages one by one to extract the links of companies.

        Yields:
            request
        """
        # iterate thourgh goals 1-17
        pages = range(1, 1630)
        # pages = [1629]
        for page in pages:

            # create url and trigger request
            url = 'https://www.unglobalcompact.org/what-is-gc/participants?page=' + str(page)
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                cb_kwargs={
                },
            )

    def parse(self, response, **kwargs):
        """Get company links from list.

        Args:
            response: html-response
            **kwargs: empty

        Yields:
            requests for company links
        """
        hrefs = response.xpath("//table[@class='participants-table']/tbody//@href").extract()
        links = ["https://www.unglobalcompact.org/" + href for href in hrefs]

        for link in links:
            yield scrapy.Request(
                url=link,
                callback=self.parse_company,
                cb_kwargs={
                },
            )

    def parse_company(self, response, **kwargs):
        """Get company information the sdg that are colored.

        Args:
            response: html-response
            **kwargs: empty

        Yields:
            loader to save company sdgs
        """
        loader = ItemLoader(item=UNGlobalCompactItem(), selector=response)

        # add values to loader and trigger items
        loader.add_xpath("company_name", "//div[@class='tile-info']/span[@class='title']/text()")
        loader.add_xpath("company_website", "//a[@class='participant-website-link']/@href")
        loader.add_xpath("company_country", "//section[@class='column two-12s org-info']//text()")

        loader.add_xpath("sdgs", "//div[@class='sdg-icons']/a[substring(@class, string-length(@class) - string-length('active') +1) = 'active']/span/text()")

        yield loader.load_item()
