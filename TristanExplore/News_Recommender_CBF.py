import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class NewsRecommenderCBF:
    def __init__(self, mind_data_path):
        self.data = self._load_data(mind_data_path)

        self.corpus = self._create_corpus()
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_df=0.8, min_df=2, ngram_range=(1, 2)
        )

        self.tfidf_matrix = self._create_tfidf_matrix()

    def _load_data(self, path):

        coloumns = [
            "news_id",
            "category",
            "sub_category",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]

        try:
            data = pd.read_csv(path, sep="\t", header=None, names=coloumns)
            print(f"00 ----------> Data loaded successfully: {len(data)} records!")

            data = data.fillna("")
        except Exception as e:
            print(f"Error loading data: {e}")
            data = None

        return data

    def _create_corpus(self):
        if self.data is None:
            return []

        corpus = (self.data["title"] + " " + self.data["abstract"]).tolist()
        print(f"01 ----------> Corpus created: {len(corpus)} documents!")
        return corpus

    def _create_tfidf_matrix(self):
        if self.data is None:
            return None

        tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print(
            f"TF-IDF matrix created: {tfidf_matrix.shape[0]} documents, {tfidf_matrix.shape[1]} terms!"
        )

        print(f"02 ----------> TF-IDF matrix created!")

        return tfidf_matrix

    # Output Functions

    def print_data(self):
        print(self.data.head().to_string())

    def get_data_frame(self):
        return self.data

    def get_tdif_matrix(self):
        return self.tfidf_matrix
