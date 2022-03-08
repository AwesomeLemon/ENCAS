import numpy as np


class PredictorSubsets:
    '''
    Contains several predictors, with each operating on a subset of the input. Outputs are averaged.
    '''

    def __init__(self, predictor_class, input_sizes, alphabet, alphabet_lb, **kwargs) -> None:
        self.n_predictors = len(input_sizes)
        self.input_sizes = input_sizes
        self.predictors = []
        input_sizes_cumsum = np.cumsum(input_sizes)
        self.input_ranges = list(zip([0] + list(input_sizes_cumsum[:-1]), input_sizes_cumsum))
        alphabets = [alphabet[s:e] for s, e in self.input_ranges]
        alphabets_lb = [alphabet_lb[s:e] for s, e in self.input_ranges]
        for input_size, a, a_lb in zip(input_sizes, alphabets, alphabets_lb):
            p = predictor_class(alphabet=a, alphabet_lb=a_lb, **kwargs)
            self.predictors.append(p)

        self.name = 'rbf_ensemble_per_ensemble_member'

    def fit(self, X, y, **kwargs):
        # assume both X and y are lists with the same lengths as the number of predictors
        for i, p in enumerate(self.predictors):
            p.fit(X[i], y[i], **kwargs)

    def predict(self, X):
        X_sep = [X[:, s:e] for s, e in self.input_ranges]
        out = None
        for x, p in zip(X_sep, self.predictors):
            preds = p.predict(x)
            if out is None:
                out = preds
            else:
                out += preds
        out /= len(self.predictors)
        return out