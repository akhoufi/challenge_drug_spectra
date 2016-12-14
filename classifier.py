from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_estimators = 100
        self.criterion='gini'
        self.max_features=20
        self.max_depth=10
        self.n_components = 100
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)),
            ('clf', RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_features=self.max_features,
                                           max_depth=self.max_depth, random_state=42))
        ])

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
