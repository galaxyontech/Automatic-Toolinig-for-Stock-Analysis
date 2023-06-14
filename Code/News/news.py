from newspaper import Article


class New:
    news_id: str
    title: str
    media: str
    date: str
    datetime: str
    description: str
    link: str
    image_link: str
    article: Article

    def __init__(self, title=None, media=None, date=None, datetime=None, description=None, link=None, image_link=None,
                 text=None, article=None) -> None:
        self.title = title
        self.media = media
        self.date = date
        self.datetime = datetime
        self.description = description
        self.link = link
        self.image_link = image_link
        self.news_id = ""
        self.text = text

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
