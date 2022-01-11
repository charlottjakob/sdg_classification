# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import TakeFirst, MapCompose, Join


def remove_trash(value):
    # delete paragraphs, and spaces at the beginning and at the end
    value = value.replace("\n","").strip()

    # if the value is not empty return it
    if value is not "":
        return value
    
    pass

class UNItem(scrapy.Item):
    # define the fields for your item here like:
    goal = scrapy.Field()
    target_title = scrapy.Field()
    target_title = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=TakeFirst(),
    )
    target_text = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=TakeFirst(),
    )
    indicator_title = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=TakeFirst(),
    )
    indicator_text = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=TakeFirst(),
    )
    info_text = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=Join(separator=" "),
    )
    related_topic_text = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=Join(separator=" "),
    )
    pass


class ScholarItem(scrapy.Item):
    # define the fields for your item here like:
    goal = scrapy.Field()
    pdf_header = scrapy.Field()
    pdf_text = scrapy.Field(
        input_processor=MapCompose(
            remove_trash,
        ),
        output_processor=Join(separator=" "),
    )
    pass