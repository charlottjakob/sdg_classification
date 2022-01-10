import scrapy
from tutorial.tutorial.items import UNItem
from scrapy.loader import ItemLoader


class UNSpider(scrapy.Spider):
    name = 'un_spider'
    allowed_domains = ['sdgs.un.org']
    custom_settings = {
        "ITEM_PIPELINES": {
            "tutorial.tutorial.pipelines.UNPipeline": 300,
        },
        "FEEDS": {
            "files/un_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):
        
        # iterate thourgh goals 1-17
        goal_numbers = range(1,18)
        for goal_number in goal_numbers:

            # create url and trigger request
            url = 'https://sdgs.un.org/goals/goal' + str(goal_number)
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                cb_kwargs={
                },
            )

    def parse(self, response):
        
        goal = response.url[30:]
        
        ## get target and indicator-texts

        # loop through targets and save titles and texts
        targets = response.xpath("//div[@class='card goal-target col-md-12']")
        for target in targets:

            # extract title and text of target
            target_title = target.xpath(".//h4[@class='goal-title']//text()").extract()
            target_text = target.xpath(".//p[@class='goal-text']//text()").extract()
        
            # loop through indicators of the target and save title and text of indicators
            indicators = target.xpath("//div[@class='goal-indicator']")
            for indicator in indicators:

                # extract title and text of indicator
                indicator_title = indicator.xpath(".//h5[@class='goal-title']//text()").extract()
                indicator_text = indicator.xpath(".//div[@class='clearfix text-formatted field field--name-description field--type-text-long field--label-hidden field__item']//text()").extract()

                # initialize loader
                loader = ItemLoader(item=UNItem(), selector=response)
                
                # add values to loader and trigger items
                loader.add_value("goal", goal)
                loader.add_value("target_title", target_title)
                loader.add_value("target_text", target_text)
                loader.add_value("indicator_title", indicator_title)
                loader.add_value("indicator_text", indicator_text)

                # trigger pipeline
                yield loader.load_item()


        # get links of related topics and trigger request if relevant



