"""Scrape companies from un global compact webpage and the sdgs they commited to."""

import scrapy
from scraping.scraping.items import UNGlobalCompactItem, ZipfilesItem

from scrapy.loader import ItemLoader
from urllib.parse import urlencode

from datetime import datetime

API_KEY = 'daea9d578cb043e3f7b09a4e9438a4c9'
# API_KEY2 = 'add727bafdc44eb9e65d0135d19b40ea'

def get_scraperapi_url(url):
    payload = {'api_key': API_KEY, 'url': url}
    proxy_url = 'http://api.scraperapi.com/?' + urlencode(payload)
    return proxy_url

def try_parsing_date(text):
    for fmt in ('%Y-%m-%d', '%d-%b-%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found for: ' + text)

class UNGlobalCompactSpider(scrapy.Spider):
    """Spider to scrapy companies and ther commited sdgs from ungloablcompact."""

    name = 'un_global_compact_spider'
    custom_settings = {
        "ITEM_PIPELINES": {
            'scraping.scraping.pipelines.MongoPipeline': 100, # we first run upload to mongodb because DownloadPdfPipeline does not return item
            'scraping.scraping.pipelines.DownloadPdfPipeline': 200,
        },
        "FEEDS": {
            "data/un_global_compact_data.csv": {"format": "csv"},
        },
        "FILES_STORE": r'/Users/charlott/Nextcloud/Documents/SRs',
        "DOWNLOAD_DELAY": 1,
        "MONGO_URI": 'mongodb+srv://charlottjakob:zjaTfgv3uVPUrEyO@cluster0.3wyread.mongodb.net/?retryWrites=true&w=majority',
        "MONGO_DB":'db_SDG'
    }

    def start_requests(self):
        """Request all pages one by one to extract the links of companies.

        Yields:
            request
        """
        # iterate thourgh the pages form last to first
        pages = list(reversed(range(0,1))) #(1500,1846
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
        countries = response.xpath("//table[@class='participants-table']/tbody//td[@class='country']/text()").extract()
        sectors = response.xpath("//table[@class='participants-table']/tbody//td[@class='sector']/text()").extract()
        types = response.xpath("//table[@class='participants-table']/tbody//td[@class='type']/text()").extract()

        # scrape additionaly famous companies
        links = ['https://unglobalcompact.org/what-is-gc/participants/2864-Deutsche-Bahn-AG','https://unglobalcompact.org/what-is-gc/participants/9429-TotalEnergies', 'https://unglobalcompact.org/what-is-gc/participants/9920-Vattenfall-AB', 'https://unglobalcompact.org/what-is-gc/participants/9195-The-Coca-Cola-Company', 'https://unglobalcompact.org/what-is-gc/participants/93111-Bonava-AB', 'https://unglobalcompact.org/what-is-gc/participants/6584-Microsoft-Corporation', 'https://unglobalcompact.org/what-is-gc/participants/152373-NVIDIA-Corporation', 'https://unglobalcompact.org/what-is-gc/participants/10041-Volkswagen-AG', 'https://unglobalcompact.org/what-is-gc/participants/8467-Siemens-AG', 'https://unglobalcompact.org/what-is-gc/participants/497-Allianz-SE', 'https://unglobalcompact.org/what-is-gc/participants/150965-Daimler-Truck-AG','https://unglobalcompact.org/what-is-gc/participants/2871-Deutsche-Telekom-AG', 'https://unglobalcompact.org/what-is-gc/participants/1194-BASF-SE', 'https://unglobalcompact.org/what-is-gc/participants/6524-Merck-KGaA', 'https://unglobalcompact.org/what-is-gc/participants/2869-Deutsche-Post-DHL-Group','https://unglobalcompact.org/what-is-gc/participants/1372-BMW-AG']
        types = ['Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company', 'Company']
        sectors = ['Industrial Transportation', 'Oil, gas, & coal', 'Gas, Water & Multiutilities', 'Beverages', 'Real Estate Investment & Services development', 'Software & Computer Services', 'Technology Hardware & Equipment', 'Automobiles & Parts', 'Technology Hardware & Equipment', 'Finance and credit services','Automobiles & Parts', 'Telecommunications equipment; telecommunications service providers', 'Chemicals', 'Pharmaceuticals & Biotechnology', 'Industrial Transportation', 'Automobiles & Parts']
        countries = ['Germany', 'France', 'Sweden', 'United States of America', 'Sweden', 'United States of America', 'United States of America', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany']



        for idx, link in enumerate(links):
            yield scrapy.Request(
                url=get_scraperapi_url(link),
                callback=self.parse_company,
                cb_kwargs={
                    "company_sector": sectors[idx] if len(sectors) == len(links) else "",
                    "company_type": types[idx] if len(types) == len(links) else "",
                    "company_country": countries[idx] if len(countries) == len(links) else "",
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
        sdgs = response.xpath("//div[@class='sdg-icons']/a[substring(@class, string-length(@class) - string-length('active') +1) = 'active']/span/text()").extract()
        documents = response.xpath("//table[@class='table-embedded']//td[@class='title']/a")
        publication_dates = response.xpath("//table[@class='table-embedded']//td[@class='published-on']/text()").extract()

        if documents:

            # loop through documents and
            for i in range(len(documents)):

                document_publication_date = try_parsing_date(publication_dates[i])

                if document_publication_date.year > 2015:

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
                                "company_country": kwargs["company_country"],
                                "sdgs": sdgs,
                                "title": title,
                                "company_sector": kwargs["company_sector"],
                                "company_type": kwargs["company_type"],
                                "document_publication_date": document_publication_date
                            },
                        )

    def parse_company_document(self, response, **kwargs):
        

        document_info = response.xpath("//section[@class='main-content-body']//text()").extract()
        links = response.xpath("//section[@class='main-content-body']//dd//a/@href").extract()
        pdf_links = [link for link in links if '.pdf' in link]
        document_reporting_period = response.xpath("//section[@class='main-content-body']//dd[preceding::dt[text()='Time period']]/ul/li/text()").get()

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
                loader.add_value("company_sector", kwargs["company_sector"])
                loader.add_value("company_type", kwargs["company_type"])
                loader.add_value("sdgs", kwargs["sdgs"])
                loader.add_value("document_link", document_link)
                loader.add_value("document_file_name", document_link.split("?")[-1])
                loader.add_value("document_publication_date", kwargs["document_publication_date"])
                loader.add_value("document_reporting_period", document_reporting_period)
                loader.add_value('file_urls', [document_link])

                yield loader.load_item()
