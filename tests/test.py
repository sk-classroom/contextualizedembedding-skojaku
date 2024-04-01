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

        df = pd.read_csv(f"{self.root}/data/eval_test.csv")
        labels = df["label"].values
        K = len(set(labels))
        X = df.drop(columns=["label"]).values
        clf = LinearDiscriminantAnalysis(n_components=K - 1).fit(X, labels)
        score = clf.score(X, labels)
        assert score > 0.5


if __name__ == "__main__":
    unittest.main()

# %%
