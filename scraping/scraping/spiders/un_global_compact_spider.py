"""Scrape companies from un global compact webpage and the sdgs they commited to."""

import scrapy
from scraping.scraping.items import UNGlobalCompactItem, ZipfilesItem
from scrapy.loader import ItemLoader


class UNGlobalCompactSpider(scrapy.Spider):
    """Spider to scrapy companies and ther commited sdgs from ungloablcompact."""

    name = 'un_global_compact_spider'
    allowed_domains = ['unglobalcompact.org']
    custom_settings = {
        "ITEM_PIPELINES": {
            'scraping.scraping.pipelines.DownloadPdfPipeline': 1,
        },
        "FEEDS": {
            "files/un_global_compact_data.csv": {"format": "csv"},
        },
        "FILES_STORE": r'/Users/charlottjakob/Documents/ecoworld/sdg-classification/files/cops'
    }

    def start_requests(self):
        """Request all pages one by one to extract the links of companies.

        Yields:
            request
        """
        # iterate thourgh goals 1-17
        pages = list(reversed(range(1200, 1630)))
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
        # loader = ItemLoader(item=UNGlobalCompactItem(), selector=response)

        # # add values to loader and trigger items
        # loader.add_xpath("company_name", "//div[@class='tile-info']/span[@class='title']/text()")
        # loader.add_xpath("company_website", "//a[@class='participant-website-link']/@href")
        # loader.add_xpath("company_country", "//section[@class='column two-12s org-info']//text()")

        # loader.add_xpath("sdgs", "//div[@class='sdg-icons']/a[substring(@class, string-length(@class) - string-length('active') +1) = 'active']/span/text()")

        company_name = response.xpath("//div[@class='tile-info']/span[@class='title']/text()").extract()
        company_website = response.xpath("//a[@class='participant-website-link']/@href").extract()
        company_country = response.xpath("//section[@class='column two-12s org-info']//text()").extract()
        sdgs = response.xpath("//div[@class='sdg-icons']/a[substring(@class, string-length(@class) - string-length('active') +1) = 'active']/span/text()").extract()

        documents = response.xpath("//table[@class='table-embedded']//td[@class='title']/a")

        if documents:
            title = documents[0].xpath("./text()").get()
            href = documents[0].xpath("./@href").get()
            document_info_link = "https://www.unglobalcompact.org/" + href

            if 'Communication on Progress' in title:

                # create url and trigger request
                yield scrapy.Request(
                    url=document_info_link,
                    callback=self.parse_company_document,
                    cb_kwargs={
                        "company_name": company_name,
                        "company_website": company_website,
                        "company_country": company_country,
                        "sdgs": sdgs,
                    },
                )


    def parse_company_document(self, response, **kwargs):

        document_info = response.xpath("//section[@class='main-content-body']//text()").extract()
        links = response.xpath("//section[@class='main-content-body']//dd//a/@href").extract()
        pdf_links = [link for link in links if '.pdf' in link]
        document_link = "https:" + pdf_links[0]

        if 'Stand alone document' in str(document_info):

            loader = ItemLoader(item=UNGlobalCompactItem(), selector=response)

            # add values to loader and trigger items
            loader.add_value("company_name", kwargs["company_name"])
            loader.add_value("company_website", kwargs["company_website"])
            loader.add_value("company_country", kwargs["company_country"])
            loader.add_value("sdgs", kwargs["sdgs"])
            loader.add_value("cop_link", document_link)
            loader.add_value("document_file_name", document_link.split("?")[-1])
            # loader.add_value('file_urls', [document_link])

            yield loader.load_item()
