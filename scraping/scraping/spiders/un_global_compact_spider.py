"""Scrape companies from un global compact webpage and the sdgs they commited to."""

import scrapy
from scraping.scraping.items import UNGlobalCompactItem, ZipfilesItem
from scrapy.loader import ItemLoader
from urllib.parse import urlencode

API_KEY = 'daea9d578cb043e3f7b09a4e9438a4c9'
# API_KEY2 = 'add727bafdc44eb9e65d0135d19b40ea'

def get_scraperapi_url(url):
    payload = {'api_key': API_KEY, 'url': url}
    proxy_url = 'http://api.scraperapi.com/?' + urlencode(payload)
    return proxy_url


class UNGlobalCompactSpider(scrapy.Spider):
    """Spider to scrapy companies and ther commited sdgs from ungloablcompact."""

    name = 'un_global_compact_spider'
    # allowed_domains = ['unglobalcompact.org']  # needs to be commented because of scraperapi
    custom_settings = {
        "ITEM_PIPELINES": {
            'scraping.scraping.pipelines.DownloadPdfPipeline': 1,
        },
        "FEEDS": {
            "data/un_global_compact_data_500_1500.csv": {"format": "csv"},
        },
        "FILES_STORE": r'/Users/charlottjakob/Documents/github_repos/sdg_classification/data/sustainability_reports_500_1500',
        "DOWNLOAD_DELAY": 1,
    }

    def start_requests(self):
        """Request all pages one by one to extract the links of companies.

        Yields:
            request
        """
        # iterate thourgh the pages form last to first
        pages = list(reversed(range(752, 900)))
        # pages = [1629]
        for page in pages:

            # create url and trigger request
            url = 'https://www.unglobalcompact.org/what-is-gc/participants?page=' + str(page)
            yield scrapy.Request(
                url=get_scraperapi_url(url),
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
                url=get_scraperapi_url(link),
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
        terms = ['Communication on Progress', 'COP', 'report', 'Report']

        company_name = response.xpath("//div[@class='tile-info']/span[@class='title']/text()").extract()
        company_website = response.xpath("//a[@class='participant-website-link']/@href").extract()
        company_country = response.xpath("//section[@class='column two-12s org-info']//text()").extract()
        sdgs = response.xpath("//div[@class='sdg-icons']/a[substring(@class, string-length(@class) - string-length('active') +1) = 'active']/span/text()").extract()
        documents = response.xpath("//table[@class='table-embedded']//td[@class='title']/a")

        if documents:

            # if min 3 documents are available take 3
            if len(documents) >= 3:
                n_documents = 3

            # if less documents, take all ot them
            else:
                n_documents = len(documents)

            # loop through documents and
            for i in range(n_documents):

                title = documents[i].xpath("./text()").get()
                href = documents[i].xpath("./@href").get()
                document_info_link = "https://www.unglobalcompact.org/" + href

                if any(term in title for term in terms):
                    # create url and trigger request
                    yield scrapy.Request(
                        url=get_scraperapi_url(document_info_link),
                        callback=self.parse_company_document,
                        cb_kwargs={
                            "company_name": company_name,
                            "company_website": company_website,
                            "company_country": company_country,
                            "sdgs": sdgs,
                            "title": title,

                        },
                    )

    def parse_company_document(self, response, **kwargs):
        

        document_info = response.xpath("//section[@class='main-content-body']//text()").extract()
        links = response.xpath("//section[@class='main-content-body']//dd//a/@href").extract()
        pdf_links = [link for link in links if '.pdf' in link]

        if pdf_links:
            document_link = "https:" + pdf_links[0]

            # If company update is Communication on Progress, the document is sutstainaiblity report if it's declared as 'Part of a' report
            if 'Communication on Progress' in kwargs["title"] or 'COP' in kwargs["title"]:
                if 'Part of a' in str(document_info):
                    is_report = True
                else:
                    is_report = False

            # if it is a report and the document can be taken regardless of the format
            elif 'report' in kwargs["title"] or 'Report' in kwargs["title"]:
                is_report = True

            # if it's something else, it's skiped
            else:
                is_report = False

            # download document if it's a report
            if is_report is True:

                loader = ItemLoader(item=UNGlobalCompactItem(), selector=response)

                # add values to loader and trigger items
                loader.add_value("company_name", kwargs["company_name"])
                loader.add_value("company_website", kwargs["company_website"])
                loader.add_value("company_country", kwargs["company_country"])
                loader.add_value("sdgs", kwargs["sdgs"])
                loader.add_value("document_link", document_link)
                loader.add_value("document_file_name", document_link.split("?")[-1])
                loader.add_value('file_urls', [document_link])

                yield loader.load_item()
