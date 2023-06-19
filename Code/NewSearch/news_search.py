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


class NewSearch:
    keywords: str

    def __init__(self, keywords: List[str]) -> None:
        self.keywords = keywords
        print(self.keywords)

    def search_key_words(self) -> str:
        """This is the function to search multiple key words and then use save and clean them

        Args:
            key_words (Collection(string)): a collection of string, or list of string, or anything iterable in python with string inside

        """
        start_time = time.time()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

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
        logging.info(word + ": searching")

        # get search result as list named results
        googlenews = GoogleNews()
        googlenews.search(word)
        results = googlenews.result()  # list

        # export the downloaded news into local file system -> to be deprecated into export into a database
        self.news_export(results=results, word=word)

    def news_export(self, results: List[Any], word: str) -> None:
        directory = "../resources/news/" + word
        meta_data_path = os.path.join(
            directory, "meta_data.json")  # path to meta_data
        # path to store err message
        errpath = os.path.join(directory, "err_log.txt")
        meta_data = {}

        # if the file directory does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(0, len(results)):
            result_date = results[i]["datetime"]
            if result_date:
                if not isinstance(result_date, float):
                    results[i]["datetime"] = result_date.strftime(
                        '%Y-%m-%d %H:%M:%S.%f')
            filename = str(i) + ".json"

            meta_data[filename] = results[i]

            filepath = os.path.join(directory, filename)

            url = results[i]["link"]
            try:  # try to scrape news url, if succeed, write it into a file
                article = Article(url)
                article.download()
                article.parse()
            # if not, write the error message into a err_log file (append writing)
            except Exception as e:
                with open(errpath, "a") as f:
                    f.write(str(e) + "\n" + "--------------------------" + "\n")
                    continue  # if not exist we kip this round
            raw_input_datasource = article.text
            formatted_input_datasource = "\n".join(raw_input_datasource.split("\n\n"))  # remove the extra newlines
            input_data_instance = New(title=results[i]["title"],
                                      media=results[i]["media"],
                                      datetime=results[i]["datetime"],
                                      description=results[i]["desc"],
                                      link=results[i]["link"],
                                      image_link=results[i]["img"],
                                      text=article.text,
                                      company = word.split(" ")[0] # hard coded
                                      )
            with open(filepath, "w") as file:
                passage_output = {"datasource_id": input_data_instance.get_news_id(),
                                  "data": formatted_input_datasource}
                json.dump(passage_output, file, indent=4)

            meta_data[filename]["news_id"] = input_data_instance.get_news_id()
            print(input_data_instance)

        with open(meta_data_path, "w") as f:
            logging.info("Metadata Saving: " + word)
            json.dump(meta_data, f, indent=4)


news_instance = NewSearch(["Apple Company News"])
news_instance.search_key_words()
