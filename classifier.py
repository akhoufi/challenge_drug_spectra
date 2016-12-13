import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.params={'colsample_bytree': 0.5, 'silent': 1, 'eval_metric': 'error', 'nthread': 6, 'min_child_weight': 1.0, 'n_estimators': 690.0, 'subsample': 0.65, 'eta': 0.30000000000000004, 'objective': 'multi:softprob', 'num_class': 4, 'max_depth': 9, 'gamma': 0.65}
        self.trainRound = 100

    def fit(self, X, y):
        self.lbl_enc = preprocessing.LabelEncoder()
        self.label_to_num = self.lbl_enc.fit_transform(y)
        train_xgb = xgb.DMatrix(X, self.label_to_num)
        self.clf= xgb.train(self.params, train_xgb, self.trainRound)

    def predict(self, X):
        num_to_label= self.lbl_enc.inverse_transform(self.label_to_num)
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num_to_label[i] for i in y])
    

    def predict_proba(self, X):
        test_xgb  = xgb.DMatrix(X)
        return self.clf.predict(test_xgb)
