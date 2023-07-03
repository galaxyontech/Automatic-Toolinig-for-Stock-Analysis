from typing import Any, Union, Dict, List
from Code.News.news import New
from Code.NewsClustering.open_api import OpenAISummarization
from Code.Database.Firebase import FireBase

class Data:
    data_body: Union[list[New], Dict[str,Dict[str,str]]]

    def __init__(self, data_body: Any):
        self.data_body = data_body

    def summarize_news_data(self):
        summarized_news_data: Dict[str,str] = {}
        for new in self.data_body:
            text: str = new.get_text()
            message: str = "Summarize the following paragraph with focus on " + "Meta:\n" + text

            # Connect to OpenAPI Instance
            api_instance: OpenAISummarization = OpenAISummarization(model_name="gpt-3.5-turbo", message=message)
            summarized_result: str = api_instance.message_summarization()
            summarized_news_data[new.get_news_id()] = summarized_result
        sum_insert = FireBase()
        sum_insert.insert_summarized_data(summarized_data=summarized_news_data,company_tag='Meta')
        print(summarized_news_data)
