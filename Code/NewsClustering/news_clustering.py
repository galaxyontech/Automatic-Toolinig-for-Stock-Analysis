import json
from time import time
from typing import List, Dict, Any
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from open_api import OpenAISummarization
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import pipeline

from pathlib import Path

path = Path(os.getcwd())
DIRECTORY_PATH_VAR = ["resources", "news", "Meta"]
SUMMARIZATION_PATH_VAR = ["resources", "Summarized Datasources", "Meta"]
DIRECTORY_PATH = os.path.join(path.parent.absolute(), os.sep.join(DIRECTORY_PATH_VAR))
SUMMARIZATION_PATH = os.path.join(path.parent.parent.absolute(), os.sep.join(SUMMARIZATION_PATH_VAR))


# Firebase: News id -> Summarized Document / News Clusters nosql non structured data, documents.... Postgresql: User
# info -> phone number:str, number of stocks:int, recent searches, membership information..... sql/mysql: structured
# data

def generate_tfidf_based_matrix(documentations: List[str], matrix_type: str):
    if matrix_type == "tf_idf":
        tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(  # tfidf
            stop_words="english",
        )
        return tfidf_vectorizer.fit_transform(documentations)
    embedder = SentenceTransformer('all-mpnet-base-v2')
    bert_based_corpus_embedding = embedder.encode(documentations)
    return bert_based_corpus_embedding


class NewsClustering:
    document_matrix: Any
    summarized_content: List[str]
    ds_id_to_data_txt: Dict[str, str]
    ds_id_list: List[str]
    document_raw_txt: List[str]
    company_tag: str

    def __init__(self):
        # populate the raw datasource content list and datasource id list
        self.summarized_content = []
        self.ds_id_to_data_txt = {}
        self.ds_id_list = []
        self.document_raw_txt = []
        self.read_raw_text_documents()
        self.create_file_path_storage()

    def create_file_path_storage(self):
        company_tag = self.ds_id_list[0].split("@")[-1]
        self.company_tag = company_tag
        if not os.path.exists(SUMMARIZATION_PATH):
            os.makedirs(SUMMARIZATION_PATH)

    def read_raw_text_documents(self):
        """
        Read from raw text documents from the storage to populate the id->raw_document and list of raw documents
        :return:
        """
        num_of_files = len(os.listdir(DIRECTORY_PATH))
        for i in range(num_of_files - 2):
            raw_data_path = DIRECTORY_PATH + "/" + str(i) + ".json"  # create raw_data_path
            data_path = os.path.join(DIRECTORY_PATH, raw_data_path)
            if "json" in data_path and "meta_data.json" not in data_path:  # extract json documents and remove
                # meta_data_json
                with open(data_path, 'r+') as datasource_input:
                    data = json.load(datasource_input)
                    datasource_id = data["datasource_id"]
                    data_text = data["data"]
                    self.ds_id_to_data_txt[datasource_id] = data_text
                    self.ds_id_list.append(datasource_id)
                    self.document_raw_txt.append(data_text)

    def generate_summarized_content(self):
        """
        Generate Summarized Content out of raw documents coming from datasources by connecting to the OpenAI API
        :return:
        """
        before_sum_time = time()
        summarization_counter: int = 0

        for i in range(len(self.ds_id_list)):
            current_doc: str = self.ds_id_to_data_txt[self.ds_id_list[i]]  # get the current document
            raw_data_path: str = str(i) + ".json"
            summarized_path_to_be_stored: str = os.path.join(SUMMARIZATION_PATH, raw_data_path)
            message: str = "Summarize the following paragraph with focus on " + "Meta:\n" + current_doc

            # Connect to OpenAPI Instance
            api_instance: OpenAISummarization = OpenAISummarization(model_name="gpt-3.5-turbo", message=message)
            summarized_result: Dict[str, str] = api_instance.message_summarization()

            # Create Summarized Json format to be stored
            datasource_id_to_summarization: dict[str, str] = {
                "ds_id": self.ds_id_list[summarization_counter],
                "data_content": summarized_result["content"]
            }

            # Append to the in-memory content
            self.summarized_content.append(summarized_result["content"])

            # store into the storage # Need to Move out of For Loop for Optimization
            with open(summarized_path_to_be_stored, 'w') as storage:
                json.dump(datasource_id_to_summarization, storage, indent=4)

            summarization_counter += 1
        after_sum_time = time()
        print("Summarization Time", after_sum_time - before_sum_time)

    def populate_summarized_content(self):
        num_of_files = len(os.listdir(SUMMARIZATION_PATH))
        summarized_data = {}
        for i in range(num_of_files):
            summarization_path = os.path.join(SUMMARIZATION_PATH, str(i) + ".json")
            storage_path = os.path.join(SUMMARIZATION_PATH, summarization_path)
            with open(storage_path, 'r') as storage:
                summarized_json = json.load(storage)

                summarized_data[summarized_json['ds_id']] = summarized_json['data_content']
                self.summarized_content.append(summarized_json["data_content"].strip())

    def k_means_clustering(self):
        """
        k means clustering method of the summarized document matrix, return the clustered label back
        :return: return the clustering label back
        """
        best_k = self.k_means_get_best_score()
        km = KMeans(n_clusters=best_k, init="k-means++", random_state=1, n_init=10)
        km.fit(self.document_matrix)  # This matrix contains list of summarizations can be tfidf or bert based
        return km.labels_

    def document_clustering_k_means_in_storage(self):
        """
        Document Clustering using kmeans. The documents are extract from storage, currently it is using the file
        system as storage format, will be changed into databse
        :return: None
        """

        self.populate_summarized_content()
        # create document matrix
        document_matrix: TfidfVectorizer = generate_tfidf_based_matrix(self.document_raw_txt, matrix_type="bert")

        # skip the visualization step self.k_means_clustering_visualization(document matrix )

        self.document_matrix = document_matrix
        print(self.document_matrix)
        k_means_label_result: List[int] = self.k_means_clustering()
        document_clusters: Dict[int, List[int]] = {}
        print(k_means_label_result)
        """
        Mapped Each document to the cluster 
        """
        for i in range(len(self.summarized_content)):
            document_label = k_means_label_result[i]
            if document_label not in document_clusters:
                document_clusters[document_label] = [i]
            else:
                document_clusters[document_label].append(i)

        print(document_clusters)

        """
        Create Summarization Engine for these results. 
        """
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        for cluster, doc_labels in document_clusters.items():
            documents = ""
            for label in doc_labels:
                doc_body: str = self.summarized_content[label]
                documents += doc_body
                documents += "\n"

            # print(documents)
            length_of_token: int = len(documents.split(" "))
            max_length: int = min(length_of_token, 300)
            min_length: int = max(int(length_of_token / 10), 1)
            generated_summary_of_clusters = summarizer(documents, max_length=max_length, min_length=min_length,
                                                       do_sample=False)
            # print(generated_summary_of_clusters)
        # self.matrix_visualization(document_matrix)

    @staticmethod
    def matrix_visualization(input_matrix: any):

        reduced_data = PCA(n_components=2).fit_transform(np.asarray(input_matrix.todense()))
        print(reduced_data)
        # print reduced_data
        fig, ax = plt.subplots()
        data_point_markings = [i for i in range(len(reduced_data))]
        for index, instance in enumerate(reduced_data):
            # print instance, index, labels[index]
            pca_comp_1, pca_comp_2 = reduced_data[index]
            ax.scatter(pca_comp_1, pca_comp_2)
        for i in range(len(reduced_data)):
            ax.annotate(data_point_markings[i], (reduced_data[i][0], reduced_data[i][1]))
        plt.show()

    def k_means_get_best_score(self):
        """
        get the best k for the clusters when doing kmeans
        :return:
        """
        silhouette_coefficients = []
        SSE = []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, init="k-means++", max_iter=500, random_state=1, n_init=10)
            km.fit(self.document_matrix)
            score = silhouette_score(self.document_matrix, km.labels_)
            silhouette_coefficients.append(score)
            SSE.append(km.inertia_)
        return silhouette_coefficients.index(max(silhouette_coefficients)) + 2


test_clustering = NewsClustering()
# test_clustering.generate_summarized_content()

test_clustering.document_clustering_k_means_in_storage()
