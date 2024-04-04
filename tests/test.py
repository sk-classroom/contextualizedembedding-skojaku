# %%
import unittest
import numpy as np
import sys
import pandas as pd
from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class TestContextualEmbedding(unittest.TestCase):

    def setUp(self):
        self.root = "."

    def test_contextual_embedding(self):

        df = pd.read_csv(f"{self.root}/data/eval_test_01.csv")
        labels = df["label"].values
        K = len(set(labels))
        X = df.drop(columns=["label"]).values
        clf = LinearDiscriminantAnalysis(n_components=K - 1).fit(X, labels)
        score = clf.score(X, labels)
        assert score > 0.5

    def test_non_contextual_embedding(self):
        df = pd.read_csv(f"{self.root}/data/eval_test_01.csv")
        labels = df["label"].values
        K = len(set(labels))
        X = df.drop(columns=["label"]).values
        clf = LinearDiscriminantAnalysis(n_components=K - 1).fit(X, labels)
        score = clf.score(X, labels)
        assert score > 0.5, f"Complete the assignment 01"

        df = pd.read_csv(f"{self.root}/data/eval_test_02.csv")
        labels = df["label"].values
        K = len(set(labels))
        X = df.drop(columns=["label"]).values
        clf = LinearDiscriminantAnalysis(n_components=K - 1).fit(X, labels)
        score2 = clf.score(X, labels)
        assert (
            score > score2
        ), f"The non-contextual embedding should be less discriminative than the contextual embedding."


if __name__ == "__main__":
    unittest.main()

# %%
