import os
from typing import Dict

import openai

openai.api_key = ""


class OpenAISummarization:
    """
    Description: This is the primary entry point for the datasource to interact with openai to get summarized
    information
    """

    message: str
    model_name: str

    def __init__(self, model_name: str, message: str) -> None:
        self.message = message
        self.model_name = model_name

    def message_summarization(self) -> Dict[str,str]:
        """
        This function return the summarization version of the input datasource.
        :return:
        """

        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self.message}
            ]
        )
        return completion.choices[0].message



