import re
from typing import Dict

from newspaper import Article
from Code.util.database_util.database_util import str_path_util


class New:
    news_id: str
    title: str
    media: str
    date: str
    datetime: str
    description: str
    link: str
    image_link: str
    company: str

    def __init__(self, title=None, media=None, date=None, datetime=None, description=None, link=None, image_link=None,
                 text=None, company=None) -> None:
        self.title = title
        self.media = media
        self.date = date
        self.datetime = datetime
        self.description = description
        self.link = link
        self.image_link = image_link
        self.company = company
        self.news_id = self.generate_news_id()  # create a unique news id for the news
        self.text = text

    def generate_news_id(self):
        clean_title = str_path_util(self.title)
        clean_media = str_path_util(self.media)
        info = [clean_title, clean_media, self.company]
        return "@".join(info)

    def get_news_id(self):
        return self.news_id

    def get_text(self):
        return self.text

    def get_article_representation(self) -> Article:
        """
        return an the article representation of the current article
        :return:
        """
        return self.article

    def get_metadata_representation(self):
        """
        return a metadata representation of the current article to understand its data
        :return:
        """
        return self.__str__()

    def __str__(self) -> str:
        string_format = f"""
        news_id = {self.news_id}
        title = {self.title}
        media = {self.media}
        date = {self.date}
        datetime = {self.datetime}
        description = {self.description}
        link = {self.link}
        image_link = {self.image_link}
        """
        return string_format

    def to_dict(self):
        dict_format = {"news_id": self.news_id, "title": self.title, "media": self.media, "date": self.date,
                       "datetime": str(self.datetime), "description": self.description, "link": self.link,
                       "image_link": self.image_link, "company": self.company}

        return dict_format
