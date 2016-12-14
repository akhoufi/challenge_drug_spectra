import numpy as np
# import pandas as pd
from sklearn.preprocessing import scale, normalize

class FeatureExtractorClf():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        XX = normalize(XX)
        XX = scale(XX, with_std=False)
        return XX
