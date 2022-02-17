"""Items for project scraping."""

import scrapy
from itemloaders.processors import TakeFirst, MapCompose, Join
import regex as re


def remove_trash(value):
    """Remove paragraph marks and dubble spaces.

    Args:
        value: to be processed

    Returns:
        value if it includes information
    """
    # delete paragraphs, and spaces at the beginning and at the end
    value = value.replace("\n", "").replace("  ", " ").strip()

    # if the value is not empty return it
    if value != "":
        return value

    pass


def extract_sdg(value):
    """Extract the number of the goal from string.

    Args:
        value: to be processed

    Returns:
        sdg number as string
    """
    return value.replace("\n", "").replace('Goal', '').replace(' ', '')


def extract_paragraphs(value):
    """Extract word chains with more than 10 unicode letters and with .!?.

    Args:
        value: to be processed

    Returns:
        paragraph
    """
    strings = re.findall(r"[\s\,\p{L}]{10,}\s[\s\,\p{L}\'\"]*[.!?]", value)

    for string in strings:
        return strings


class UNItem(scrapy.Item):
    """Item for UNSpider."""

    # define the fields for your item here like:
    goal = scrapy.Field()
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
    """Item for ScholarSpider."""

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


class TakeCountry:
    """Output-processor for company_country in UNGlobalCompactItem."""

    # nothing happens - just magic
    def __call__(self, values):
        """Extract the element after Country:.

        Args:
            values: to be processed

        Returns:
            country
        """
        for i, element in enumerate(values):
            if 'Country' in element:
                country = values[i + 1]
                return country


class UNGlobalCompactItem(scrapy.Item):
    """Item for UNGlobalCompactSpider."""

    # define the fields for your item here like:
    company_name = scrapy.Field()
    company_website = scrapy.Field()
    company_country = scrapy.Field(
        input_processor=MapCompose(remove_trash),
        output_processor=TakeCountry(),
    )
    sdgs = scrapy.Field(
        input_processor=MapCompose(extract_sdg),
        output_processor=Join(separator=", "),
    )
    pass

class WebsitesItem(scrapy.Item):
    """Item for WebsitesSpider."""

    # define the fields for your item here like:
    company_name = scrapy.Field()
    company_website = scrapy.Field()
    content_url = scrapy.Field()
    content = scrapy.Field(
        input_processor=MapCompose(remove_trash, extract_paragraphs),
        output_processor=Join(separator=" "),
    )
    pass