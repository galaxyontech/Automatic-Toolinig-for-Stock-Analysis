import logging
import threading
import time
from GoogleNews import GoogleNews
from newspaper import Article
from typing import Any, List
import os
import json
import logging
from Code.News.news import New
from newspaper import Config
from Code.Database.Firebase import FireBase


class NewSearch:
    keywords: List[str]

    def __init__(self, keywords: List[str]) -> None:
        self.keywords = keywords

    def search_key_words(self) -> None:
        """This is the function to search multiple keywords and then use save and clean them

        Args:
            key_words (Collection(string)): a collection of string, or list of string, or anything iterable in python with string inside

        """
        start_time = time.time()
        thread_list = []
        """ for multi threading cases"""

        def search_and_clean(word):
            self.news_download(word)

        for key_word in self.keywords:
            t = threading.Thread(target=search_and_clean, args=(key_word,))
            thread_list.append(t)

        for i in range(0, len(self.keywords)):
            thread_list[i].start()
            logging.info("Thread started: " + self.keywords[i])
        for i in range(0, len(self.keywords)):
            thread_list[i].join()
            logging.info("Thread finished: " + self.keywords[i])

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info("Run Time: " + str(elapsed_time) + " seconds")  # Logging

    def news_download(self, word):
        """This function will reasearch a key word and save the result into a folder named of the key word and the result as sub files named 1.txt, 2.txt, etc.

        Args:
            key_word (string): the key word for searching in google news
        """

        related_key_words = ["", "stock"]
        dedupped_results = {}
        # get search result as list named results
        for w in related_key_words:
            googlenews = GoogleNews(period='1d')
            googlenews.search(word + " " + w)
            results = googlenews.result(sort=True)  # list
            for r in results:
                dedupped_results[r['link']] = r
            googlenews.clear()
        # export the downloaded news into local file system -> to be deprecated into export into a database
        final_results = []
        for i in dedupped_results.values():
            final_results.append(i)
        sorted_final_results = sorted(final_results, key=lambda x: x['datetime'], reverse=True)
        print("----------")
        for i in sorted_final_results:
            print(i)
        self.news_export(results=sorted_final_results, word=word)

    def news_export(self, results: List[Any], word: str) -> None:
        """
        Export news from google news and turns them into the datasource format
        :param results:
        :param word:
        :return:
        """
        directory = "../resources/news/" + word
        errpath = os.path.join(directory, "err_log.txt")
        meta_data = {}
        error_meta_data = {}
        news_list = []  # list of internal news instance to be summarized using openai api

        # if the file directory does not exist

        for i in range(0, len(results)):
            result_date = results[i]["datetime"]

            # format datetime
            if result_date:
                if not isinstance(result_date, float):
                    results[i]["datetime"] = result_date.strftime(
                        '%Y-%m-%d %H:%M:%S.%f')
            else:
                results[i]["datetime"] = time.time()

            # download file from the url
            filename = str(i) + ".json"
            filepath = os.path.join(directory, filename)
            article = None
            current_exception: str = ""
            url: str = results[i]["link"]
            user_agent: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                              'Chrome/50.0.2661.102 Safari/537.36 '
            config = Config()
            config.request_timeout = 50
            config.browser_user_agent = user_agent
            try:  # try to scrape news url, if succeeded, write it into a file
                article = Article(url, config=config)
                article.download()
                article.parse()
            # if not, write the error message into an err_log file (append writing)
            except Exception as e:
                current_exception = str(e)
            assert article, "No article found for any giving url"

            # generate into internal news instance
            raw_input_datasource: str = article.text
            formatted_input_datasource: str = "\n".join(raw_input_datasource.split("\n\n"))  # remove the extra newlines
            input_data_instance: New = New(title=results[i]["title"],
                                           media=results[i]["media"],
                                           datetime=results[i]["datetime"],
                                           description=results[i]["desc"],
                                           link=results[i]["link"],
                                           image_link=results[i]["img"],
                                           text=article.text,
                                           company=word.split(" ")[0]  # hard coded
                                           )

            news_list.append(input_data_instance)

            with open(filepath, "w") as file:
                passage_output = {"datasource_id": input_data_instance.get_news_id(),
                                  "data": formatted_input_datasource}
                json.dump(passage_output, file, indent=4)

            if current_exception:
                error_meta_data[input_data_instance.news_id] = current_exception
            meta_data[input_data_instance.get_news_id()] = results[i]
            meta_data[input_data_instance.get_news_id()]["news_id"] = input_data_instance.get_news_id()

        # dumping metadata information into the database
        metadata_insert = FireBase()
        metadata_insert.insert_metadata_data(metadata=meta_data, company_tag=word)


        # dumping news instance into api summarization workflow and insert into the database
        from Code.Data.data import Data
        news_data = Data(data_body=news_list)
        news_data.summarize_news_data()


        with open(errpath, 'w') as error_input:
            json.dump(error_meta_data, error_input, indent=4)


news_instance = NewSearch(["Meta"])
news_instance.search_key_words()
