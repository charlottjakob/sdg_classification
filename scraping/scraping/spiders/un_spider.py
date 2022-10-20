"""Scrape SDG explanation texts from UN webpage."""

import scrapy
from scraping.scraping.items import UNItem
from scrapy.loader import ItemLoader


class UNSpider(scrapy.Spider):
    """Scrape SDG explanation text from UN webpage."""

    name = 'un_spider'
    allowed_domains = ['sdgs.un.org']
    custom_settings = {
        "ITEM_PIPELINES": {
            "scraping.scraping.pipelines.BasicPipeline": 300,
        },
        "FEEDS": {
            "data/un_data.csv": {"format": "csv"},
        },
    }

    def start_requests(self):
        """Start requests for 17 SDGs.

        Yields:
            scraping requests
        """
        # iterate thourgh goals 1-17
        sdg_numbers = range(1, 18)
        for sdg_number in sdg_numbers:

            # create url and trigger request
            url = 'https://sdgs.un.org/goals/goal' + str(sdg_number)
            yield scrapy.Request(
                url=url,
                callback=self.parse_goal,
                cb_kwargs={"sdg": str(sdg_number)},
            )

    def parse_goal(self, response, **kwargs):
        """Extract data, save data and request further webpages.

        Collect target name, target text, indicator name, indicator text and SDG info text
        Start requests for papers(related topics) adressing exactly the specific SDG

        Args:
            response: html body to be explored
            kwargs: specific sdg

        Yields:
            for UNItem to process text data
            scarping requests for related topics
        """
        sdg = kwargs["sdg"]

        # # get target and indicator-texts

        # loop through targets and save titles and texts
        targets = response.xpath("//div[@class='card goal-target col-md-12']")
        for target in targets:

            # extract title and text of target
            target_title = target.xpath(".//h4[@class='goal-title']//text()").extract()
            target_text = target.xpath(".//p[@class='goal-text']//text()").extract()

            # loop through indicators of the target and save title and text of indicators
            indicators = target.xpath("//div[@class='goal-indicator']")

            loader = ItemLoader(item=UNItem(), selector=response)
            loader.add_value("sdg", sdg)
            loader.add_value("target_title", target_title)
            loader.add_value("target_text", target_text)
            yield loader.load_item()

            for indicator in indicators:

                # extract title and text of indicator
                indicator_title = indicator.xpath(".//h5[@class='goal-title']//text()").extract()
                indicator_text = indicator.xpath(".//div[@class='clearfix text-formatted field field--name-description field--type-text-long field--label-hidden field__item']//text()").extract()

                # initialize loader
                loader = ItemLoader(item=UNItem(), selector=response)

                # add values to loader and trigger items
                loader.add_value("sdg", sdg)
                loader.add_value("indicator_title", indicator_title)
                loader.add_value("indicator_text", indicator_text)

                # trigger pipeline
                yield loader.load_item()

        # # extract info-texts
        # initialize loader
        loader = ItemLoader(item=UNItem(), selector=response)

        # add values to loader and trigger items
        loader.add_value("sdg", sdg)
        loader.add_xpath("info_text", "//section[substring(@class, string-length(@class) - string-length('text-formatted') +1) = 'text-formatted']//p/text()")

        # trigger pipeline
        yield loader.load_item()

        # get links of related topics and trigger request if relevant
        related_topics = response.xpath("//div[@class='card-content-topics']")
        for related_topic in related_topics:

            related_goals = related_topic.xpath(".//div[@class='label-group-content']/span/@class").extract()
            href = related_topic.xpath("./a/@href").get()

            if len(related_goals) == 1 and related_goals[0][25:] == sdg:
                url = "https://sdgs.un.org" + href
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_related_topic,
                    cb_kwargs={"sdg": sdg},
                )

    def parse_related_topic(self, response, **kwargs):
        """Collect text of relatec topic that addresses exactly one SDG.

        Args:
            response: html body to be explored
            kwargs: sdg

        Yields:
            loader for UNItem to process related_topic_text
        """
        # # extract description-texts
        # initialize loader
        loader = ItemLoader(item=UNItem(), selector=response)

        # add values to loader and trigger items
        loader.add_value("sdg", kwargs["sdg"])
        loader.add_xpath("related_topic_text", "//div[@id='description']/div[@class='row']//p/text()")
        loader.add_xpath("related_topic_text", "//ul[@class='timeline2']//div[@class='desc']/text()")
        # trigger pipeline
        yield loader.load_item()
