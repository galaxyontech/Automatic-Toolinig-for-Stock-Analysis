import json
from time import time
from typing import List, Dict, Any
import os
import numpy as np
from kneed import KneeLocator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from Code.NewsClustering.OpenAI_API import OpenAISummarization
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import pipeline

from pathlib import Path

path = Path(os.getcwd())
DIRECTORY_PATH_VAR = ["resources", "news", "Meta"]
SUMMARIZATION_PATH_VAR = ["resources", "Summarized Datasources"]
DIRECTORY_PATH = os.path.join(path.parent.absolute(), os.sep.join(DIRECTORY_PATH_VAR))
SUMMARIZATION_PATH = os.path.join(path.parent.parent.absolute(), os.sep.join(SUMMARIZATION_PATH_VAR))





class NewsClustering:
    document_matrix: Any
    summarized_content: List[str]
    raw_datasource_content_lists: Dict[str, str]
    datasource_id_list: List[str]

    def __init__(self):
        # populate the raw datasource content list and datasource id list
        self.summarized_content = []
        self.raw_datasource_content_lists = {}
        self.datasource_id_list = []
        self.read_raw_text_documents()
        self.create_file_path_storage()

    def create_file_path_storage(self):
        company_tag = self.datasource_id_list[0].split("@")[-1]
        if not os.path.exists(SUMMARIZATION_PATH):
            os.makedirs(SUMMARIZATION_PATH)
    def read_raw_text_documents(self):
        """
        Read from raw text documents from the storage to populate the id->raw_document and list of raw documents
        :return:
        """
        raw_datasource_content_lists = {}
        datasource_id_list = []
        for path in os.listdir():
        for i in range(10):
            path = DIRECTORY_PATH + "/" + str(i) + ".json"
            with open(path, 'r+') as datasource_input:
                data = json.load(datasource_input)
                datasource_id = data["datasource_id"]
                data_text = data["data"]
                raw_datasource_content_lists[datasource_id] = data_text
                datasource_id_list.append(datasource_id)
        self.raw_datasource_content_lists = raw_datasource_content_lists
        self.datasource_id_list = datasource_id_list
        print(self.raw_datasource_content_lists)
        print(self.datasource_id_list)


    def generate_summarized_content(self):
        before_sum_time = time()
        for i in range(10):
            file_path = os.sep.join([self.datasource_id_list[i].split("@")[-1], str(i) + ".json", ])
            print(file_path)
            storage_path = os.path.join(SUMMARIZATION_PATH, file_path)
            message = "Can you Summarize the following paragraph\n" + self.raw_datasource_content_lists[
                self.datasource_id_list[i]]
            # Connect to OpenAPI Instance

            api_instance = OpenAISummarization(model_name="gpt-3.5-turbo", message=message)
            summarized_result = api_instance.message_summarization()

            # Create Summarized Json format to be stored
            datasource_id_to_summarization = {
                "ds_id": self.datasource_id_list[i],
                "data_content": summarized_result["content"]
            }

            # Append to the in-memory content
            self.summarized_content.append(summarized_result["content"])

            # store into the storage
            with open(storage_path, 'w') as storage:
                json.dump(datasource_id_to_summarization, storage, indent=4)

        after_sum_time = time()
        print("Summarization Time", after_sum_time - before_sum_time)

    def document_clustering_k_means_in_memory(self):
        vectorizer = TfidfVectorizer(
            stop_words="english",
        )
        t0 = time()
        X_tfidf = vectorizer.fit_transform(self.summarized_content)

    def document_clustering_k_means_in_storage(self):
        for i in range(10):
            file_path = os.sep.join([self.datasource_id_list[i].split("@")[-1], str(i) + ".json", ])
            print(file_path)
            storage_path = os.path.join(SUMMARIZATION_PATH, file_path)
            print(storage_path)

            with open(storage_path, 'r') as storage:
                summarized_json = json.load(storage)
                self.summarized_content.append(summarized_json["data_content"].strip())

        for i in self.summarized_content:
            print(i)
            print("-------------------------")
        vectorizer = TfidfVectorizer(
            stop_words="english",
        )
        t0 = time()
        X_tfidf = vectorizer.fit_transform(self.summarized_content)
        self.document_matrix = X_tfidf
        print(f"vectorization done in {time() - t0:.3f} s")

        best_k = self.k_means_get_best_score()
        km = KMeans(n_clusters=best_k, init="k-means++", random_state=1)
        km.fit(self.document_matrix)
        document_clusters = {}

        for i in range(len(self.summarized_content)):
            document_label = km.labels_[i]
            if document_label not in document_clusters:
                document_clusters[document_label] = [i]
            else:
                document_clusters[document_label].append(i)

        print(document_clusters)

        summarizer = pipeline("summarization")
        for cluster, doc_labels in document_clusters.items():
            documents = ""
            for label in doc_labels:
                doc_body = self.summarized_content[label]
                documents += doc_body
                documents += "\n"

            print(documents)
            length_of_token = len(documents.split(" "))
            max_length = min(length_of_token, 300)
            generated_summary_of_clusters = summarizer(documents, max_length=max_length, min_length=30, do_sample=False)
            print(generated_summary_of_clusters)

    @staticmethod
    def k_means_clustering_visualization(input_matrix: any):

        reduced_data = PCA(n_components=2).fit_transform(np.asarray(input_matrix.todense()))

        # print reduced_data
        fig, ax = plt.subplots()
        for index, instance in enumerate(reduced_data):
            # print instance, index, labels[index]
            pca_comp_1, pca_comp_2 = reduced_data[index]
            ax.scatter(pca_comp_1, pca_comp_2)
        plt.show()

    def k_means_get_best_score(self):
        """
        get the best k for the clusters
        :return:
        """
        silhouette_coefficients = []
        SSE = []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, init="k-means++", max_iter=500, random_state=1)
            km.fit(self.document_matrix)
            score = silhouette_score(self.document_matrix, km.labels_)
            silhouette_coefficients.append(score)
            SSE.append(km.inertia_)
        return silhouette_coefficients.index(max(silhouette_coefficients)) + 2


test_clustering = NewsClustering()
test_clustering.generate_summarized_content()

test_clustering.document_clustering_k_means_in_storage()
