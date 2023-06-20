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
    company:str

    def __init__(self, title=None, media=None, date=None, datetime=None, description=None, link=None, image_link=None,
                 text=None, company= None) -> None:
        self.title = title
        self.media = media
        self.date = date
        self.datetime = datetime
        self.description = description
        self.link = link
        self.image_link = image_link
        self.company = company
        self.news_id = self.generate_news_id() # create a unique news id for the news
        self.text = text

    def generate_news_id(self):
        info = [self.title, self.media, str(self.datetime), self.company]
        return "@".join(info)

    def get_news_id(self):
        return self.news_id

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
