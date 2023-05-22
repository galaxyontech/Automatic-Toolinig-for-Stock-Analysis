
class New:
    news_id: str
    title:str 
    media:str
    date:str 
    datetime:str 
    description:str
    link:str
    image_link: str 

    def __init__(self,news_id, title, media, date , datetime, description, link, image_link,text) -> None:
        self.title = title
        self.media = media
        self.date = date
        self.datetime = datetime
        self.description = description
        self.link = link
        self.image_link = image_link
        self.new_id = ""
        self.text = text



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
    


