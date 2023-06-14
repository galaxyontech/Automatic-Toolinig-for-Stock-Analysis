from time import time
from typing import List

import numpy as np
import pandas as pd
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer


def clusering():
    articles = ['Data Science', 'Artificial intelligence', 'European Central Bank', 'Bank',
                'Financial technology', 'International Monetary Fund', 'Basketball', 'Swimming']
    wiki_lst = []
    title = []
    for article in articles:
        print("loading content: ", article)
        wiki_lst.append(wikipedia.page(article).content)
        title.append(article)

    vectorizer = TfidfVectorizer(
        stop_words="english",
    )
    t0 = time()
    X_tfidf = vectorizer.fit_transform(articles)
    print(type(X_tfidf))
    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")

    from sklearn.cluster import KMeans



class NewsClustering:
    documents: List[str]

    def __init__(self, documents=None, ):
        if documents is None:
            documents = []
        self.documents = documents


sample_clustering = NewsClustering()
clusering()