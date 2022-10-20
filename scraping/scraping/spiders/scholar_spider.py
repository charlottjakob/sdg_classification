import scrapy
from scrapy.loader import ItemLoader
from scraping.scraping.items import ScholarItem
from urllib.parse import urlencode
import re

API_KEY = 'daea9d578cb043e3f7b09a4e9438a4c9'


def get_scraperapi_url(url):
    payload = {'api_key': API_KEY, 'url': url}
    proxy_url = 'http://api.scraperapi.com/?' + urlencode(payload)
    return proxy_url


class ScholarSpider(scrapy.Spider):
    name = 'scholar_spider'
    start_urls = ['https://scholar.google.com/scholar?hl=en&q=SDG+Goal+1']
    custom_settings = {
        "ITEM_PIPELINES": {
            "scraping.scraping.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "data/scholar_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):

        # iterate thourgh goals 1-17
        sdg_numbers = range(1, 18)
        for sdg_number in sdg_numbers:

            # # create url and trigger request
            # urls = ['https://scholar.google.com/scholar?hl=en&q="Goal+' + str(goal_number) + '"',
            #        'https://scholar.google.com/scholar?hl=en&q="SDG+' + str(goal_number) + '"']
            # # meta = {
            # #     "proxy": "http://scraperapi:daea9d578cb043e3f7b09a4e9438a4c9@proxy-server.scraperapi.com:8001"
            # # }
            # for url in urls:
            #     yield scrapy.Request(
            #         url=get_scraperapi_url(url),
            #         callback=self.parse,
            #         cb_kwargs={"goal": str(goal_number)},
            #         # meta=meta
            #     )
            for i in range(0, 20):

                urls = ['https://scholar.google.com/scholar?start={}&q=SDG+{}&hl=en&as_sdt=0,5'.format(i * 10, str(sdg_number)),
                        'https://scholar.google.com/scholar?start={}&q=goal+{}&hl=en&as_sdt=0,5'.format(i * 10, str(sdg_number))]

                for url in urls:
                    yield scrapy.Request(
                        url=get_scraperapi_url(url),
                        callback=self.parse,
                        cb_kwargs={"sdg": str(sdg_number)},
                        # meta=meta
                    )    

    def parse(self, response, **kwargs):

        # find the first 10 search-results and trigger request if valid
        scholar_results = response.xpath("//div[@class='gs_r gs_or gs_scl']")
        for scholar_result in scholar_results:

            # extract file-type, link and header
            file_type = scholar_result.xpath(".//span[@class='gs_ct1']/text()").get()
            href = scholar_result.xpath(".//h3[@class='gs_rt']/a/@href").get()
            header = scholar_result.xpath("string(.//h3/a)").get()
            # description = scholar_result.xpath(".//div[@class='gs_rs']/text()").get()

            if href and header:

                # check if header includes sdg
                if 'sdg' in header.lower() or 'sustainable development goal' in header.lower():

                    # check if the goal_number is the only number in the header and trigger request
                    numbers = re.findall(r'[0-9]+', header)
                    if len(numbers) == 1 and numbers[0] == kwargs["sdg"]:

                        # wenn nicht PDF und nicht Buch dann trigger Abstract suche
                        if file_type != "[PDF]" and 'books.google' not in href:

                            yield scrapy.Request(
                                url=get_scraperapi_url(href),
                                callback=self.find_abstract,
                                cb_kwargs={
                                    "sdg": kwargs["sdg"],
                                    "header": header,
                                    "content_url": href,
                                },
                                # meta={'proxy': response.meta['proxy']}
                            )

        # go to next google-page
        relative_next_page = response.xpath("//td[@align='left']/a/@href").get()

        if relative_next_page:
            absolute_next_page = "https://scholar.google.com" + relative_next_page
            yield scrapy.Request(
                url=get_scraperapi_url(absolute_next_page),
                callback=self.parse,
                cb_kwargs={
                    "goal": kwargs["goal"],
                },
                # meta={'proxy': response.meta['proxy']}
            )

    def find_abstract(self, response, **kwargs):

        loader = ItemLoader(item=ScholarItem(), selector=response)

        # add values to loader and trigger items
        loader.add_value("sdg", kwargs["sdg"])
        loader.add_value("paper_header", kwargs["header"])
        loader.add_xpath("abstract", "//p/text()")
        loader.add_xpath("abstract", "//div/text()")
        loader.add_value("content_url", kwargs["content_url"])

        # trigger pipeline
        yield loader.load_item()
