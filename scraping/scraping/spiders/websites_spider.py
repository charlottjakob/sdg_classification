"""Scrape text from companies websites."""

import scrapy
from scraping.scraping.items import WebsitesItem
from scrapy.loader import ItemLoader
import re
import pandas as pd


class WebsitesSpider(scrapy.Spider):
    """Scrape text from companies websites."""

    name = 'websites_spider'
    # allowed_domains = ['unglobalcompact.org']
    custom_settings = {
        "ITEM_PIPELINES": {
            "scraping.scraping.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "files/website_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):
        """Request all company_websites from un_global_compacht_data.

        Yields:
            request
        """
        # load data
        un_global_compact_companies_all = pd.read_csv('files/un_global_compact_data.csv')

        # filter for rows with complete information
        un_global_compact_companies = un_global_compact_companies_all.dropna()

        # filter for companies with less than 5 sdgs
        un_global_compact_companies['sum_sdgs'] = un_global_compact_companies['sdgs'].apply(lambda x: len(x.split(", ")))
        un_global_compact_companies = un_global_compact_companies[un_global_compact_companies['sum_sdgs'] < 5]

        # filter for companies with language english and german
        en_de_countries = ['Germany', 'Australia', 'Ireland', 'United Kingdom', 'United States of America', 'Canada']
        un_global_compact_companies = un_global_compact_companies[un_global_compact_companies['company_country'].isin(en_de_countries)]

        # Start scraping of company_websites
        for i, company in un_global_compact_companies[:30].iterrows():
            yield scrapy.Request(
                url=company['company_website'],
                callback=self.parse,
                cb_kwargs={
                    "company_name": company['company_name'],
                    "company_website": company['company_website'],
                },
            )

    def parse(self, response, **kwargs):
        """Extract text and request further links.

        Args:
            response: html-response
            **kwargs: company_name, company_website

        Yields:
            requests scraping of company website links
        """
        # Initialize loader
        loader = ItemLoader(item=WebsitesItem(), selector=response)

        # add values and website content to loader
        loader.add_value("company_name", kwargs["company_name"])
        loader.add_value("company_website", kwargs["company_website"])
        loader.add_value("content_url", response.url)

        text_list = response.xpath("//p/text()").extract()
        for text in text_list:
            loader.add_value("content", text)

        # trigger pipeline
        yield loader.load_item()

        # #  extract links to further pages of the company_website and request
        # extract all hrefs and loop through them
        hrefs = response.xpath("//@href").extract()
        for href in hrefs:

            # if link is not complete add domain of current website
            if not href.startswith("http"):

                # extract domain from response.url
                domain_search = re.search(
                    "(^https?\:\/\/(\w{0,3}\.)?[^.]*\.[\w\.]+)\/", response.url
                )
                if domain_search:
                    domain = domain_search.group(1)

                    # add domain as the beginning of url
                    href = domain + href

            # if link on company_domain trigger request
            if kwargs["company_website"] in href:

                # if links are webpages
                if not href.endswith(('.woff', '.pdf')):
                    yield scrapy.Request(
                        url=href,
                        callback=self.parse,
                        cb_kwargs={
                            "company_name": kwargs['company_name'],
                            "company_website": kwargs['company_website'],
                        },
                    )
