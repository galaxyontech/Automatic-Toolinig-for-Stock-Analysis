
class New:
    news_id: str
    title:str 
    media:str
    date:str 
    datetime:str 
    description:str
    link:str
    image_link: str 

    def __init__(self) -> None:
        self.news_id = None 
        self.title = None  
        self.media = None 
        self.date = None  
        self.datetime = None  
        self.description = None 
        self.link = None 
        self.image_link = None  

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
    


