"""Feature fusion and calibrated classifier, shared by training and serving."""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASELINE = {"C": 10.0, "weight": 1.0, "ngram_range": (2, 4), "min_df": 3}


class HybridModel:
    def __init__(self, C=10.0, weight=1.0, ngram_range=(2, 4), min_df=3):
        self.params = dict(C=C, weight=weight, ngram_range=ngram_range, min_df=min_df)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                                        min_df=min_df, max_features=150000)
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(C=C, max_iter=2000, random_state=42)
        self.calibrator = None

    def _fuse(self, texts, embeddings):
        dense = self.scaler.transform(embeddings) * (self.params["weight"] / np.sqrt(768))
        return hstack([self.vectorizer.transform(texts), csr_matrix(dense)]).tocsr()

    def fit(self, texts, embeddings, labels):
        if embeddings.shape != (len(texts), 768) or not np.isfinite(embeddings).all():
            raise ValueError("Expected finite 768-dimensional MuRIL embeddings.")
        self.vectorizer.fit(texts)
        self.scaler.fit(embeddings)
        self.classifier.fit(self._fuse(texts, embeddings), labels)
        return self

    def raw_proba(self, texts, embeddings):
        return self.classifier.predict_proba(self._fuse(texts, embeddings))

    def calibrate(self, texts, embeddings, labels):
        scores = np.log(np.clip(self.raw_proba(texts, embeddings), 1e-6, 1))
        self.calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(scores, labels)
        return self

    def predict_proba(self, texts, embeddings):
        p = self.raw_proba(texts, embeddings)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(np.log(np.clip(p, 1e-6, 1)))
        return p
