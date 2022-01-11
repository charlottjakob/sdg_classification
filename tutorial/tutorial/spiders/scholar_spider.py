import scrapy
from scrapy.loader import ItemLoader
from tutorial.tutorial.items import ScholarItem
import PyPDF2
import io
import re


class ScholarSpider(scrapy.Spider):
    name = 'scholar_spider'
    start_urls = ['https://scholar.google.com/scholar?hl=en&q=SDG+Goal+1']
    custom_settings = {
        "ITEM_PIPELINES": {
            "tutorial.tutorial.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "files/scholar_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):
        
        # iterate thourgh goals 1-17
        goal_numbers = range(1,18)
        for goal_number in goal_numbers:

            # create url and trigger request
            urls = ['https://scholar.google.com/scholar?hl=en&q="Goal+' + str(goal_number) + '"',
                   'https://scholar.google.com/scholar?hl=en&q="SDG+' + str(goal_number) + '"']
            
            for url in urls:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse,
                    cb_kwargs={"goal": str(goal_number),
                    },
                )


    def parse(self, response, **kwargs):

        # find the first 10 search-results and trigger request if valid
        scholar_results = response.xpath("//div[@class='gs_r gs_or gs_scl']")
        for scholar_result in scholar_results:
            
            # extract file-type, link and header
            file_type = scholar_result.xpath(".//span[@class='gs_ctg2']/text()").get()
            href = scholar_result.xpath(".//div[@class='gs_or_ggsm']/a/@href").get()
            header = scholar_result.xpath("string(.//h3/a)").get()
            
            # check if pdf-file is available for paper and trigger request if valid
            if file_type and href and header and file_type == "[PDF]":
                
                # check if header includes sdg
                if 'sdg' in header.lower() or 'sustainable development goal' in header.lower():
                    
                    # check if the goal_number is the only number in the header and trigger request
                    numbers = re.findall(r'[0-9]+', header)
                    if len(numbers) == 1 and numbers[0] == kwargs["goal"]:

                        yield scrapy.Request(
                            url=href,
                            callback=self.parse_pdf,
                            cb_kwargs={
                                "goal": kwargs["goal"],
                                "pdf_header": header,
                            },
                        )          

    def parse_pdf(self, response, **kwargs):

        reader = PyPDF2.PdfFileReader(io.BytesIO(response.body))
        
        amount_of_pages = reader.getNumPages()
        pdf_text = ""
        for i in range(amount_of_pages):
            
            page = reader.getPage(i)
            page_content = page.extractText()

            page_strings = page_content.split('\n')
            page_long_strings = [string for string in page_strings if len(string)> 50]
            page_text = " ".join(page_long_strings)

            pdf_text = pdf_text + " " + page_text

        if pdf_text != "":
            # initialize loader
            loader = ItemLoader(item=ScholarItem(), selector=response)
            
            # add values to loader and trigger items
            loader.add_value("goal", kwargs["goal"])
            loader.add_value("pdf_header", kwargs["pdf_header"] )
            loader.add_value("pdf_text", pdf_text)

            # trigger pipeline
            yield loader.load_item()

