
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureBuilder:
    def __init__(self, config):
        self.config = config
        self.tfidf = None
        self.embedding_model = None

    def fit_tfidf(self, corpus):
        max_features = self.config.get("features.text.tfidf_max_features", 50000)
        ngram_range = tuple(self.config.get("features.text.tfidf_ngram_range", [1, 2]))

        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
        )
        matrix = self.tfidf.fit_transform(corpus)
        return matrix

    def transform_tfidf(self, corpus):
        if self.tfidf is None:
            raise ValueError("TF-IDF not fitted")
        return self.tfidf.transform(corpus)

    def fit_embeddings(self, corpus, batch_size=64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed")

        model_name = self.config.get("features.text.embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)

        embeddings = self.embedding_model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def get_item_index(self, items):
        return pd.Series(items.index.values, index=items["item_id"]).drop_duplicates()
