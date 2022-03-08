"""
Implementation based on the one provided by the NAT team, their original comment below:

The Ensemble scheme is based on the implementation from:
https://github.com/yn-sun/e2epp/blob/master/build_predict_model.py
https://github.com/HandingWang/RF-CMOCO
"""

import numpy as np
from acc_predictor.rbf import RBF


class RBFEnsemble:

    def __init__(self, ensemble_size=500, alphabet=None, alphabet_lb=None, **kwargs) -> None:
        self.n_models = ensemble_size
        self.verbose = True
        self.alphabet = alphabet
        self.alphabet_lb = alphabet_lb
        self.name = 'rbf_ensemble'
        self.models = None
        self.features = None
        self.model_predictions = np.zeros(self.n_models)

    def fit(self, X, y, **kwargs):
        n, m = X.shape
        features = []
        models = []

        if self.verbose:
            print(f"Constructing RBF ensemble surrogate model with sample size = {n}, ensemble size = {self.n_models}")

        for i in range(self.n_models):
            sample_idx = np.arange(n)
            np.random.shuffle(sample_idx)
            X = X[sample_idx, :]
            y = y[sample_idx]

            feature_idx = np.arange(m)
            np.random.shuffle(feature_idx)
            n_feature = np.random.randint(1, m + 1)
            selected_feature_ids = feature_idx[0:n_feature]
            X_selected = X[:, selected_feature_ids]
            # rbf fails if there are fewer training points than features => check & resample if needed
            idx_unique = np.unique(X_selected, axis=0, return_index=True)[1]
            while len(idx_unique) <= n_feature or len(idx_unique) == 1:
                feature_idx = np.arange(m)
                np.random.shuffle(feature_idx)
                n_feature = np.random.randint(1, m + 1)
                selected_feature_ids = feature_idx[0:n_feature]
                X_selected = X[:, selected_feature_ids]
                idx_unique = np.unique(X_selected, axis=0, return_index=True)[1]

            features.append(selected_feature_ids)
            rbf = RBF(kernel='cubic', tail='linear',
                      alphabet=self.alphabet[selected_feature_ids], alphabet_lb=self.alphabet_lb[selected_feature_ids])
            rbf.fit(X_selected, y)
            models.append(rbf)

        if self.models is not None:
            del self.models
        if self.features is not None:
            del self.features

        self.models = models
        self.features = features

    def predict(self, X):
        n = len(X)
        y = np.zeros(n)

        for i in range(n):
            this_test_data = X[i, :]

            for j, (rbf, feature) in enumerate(zip(self.models, self.features)):
                self.model_predictions[j] = rbf.predict(this_test_data[feature][np.newaxis, :])[0]

            y[i] = np.nanmedian(self.model_predictions)

        return y[:, None]
