import numpy as np


class PredictorContainer:
    '''
    Contains several predictors
    '''

    def __init__(self, predictors, name, **kwargs) -> None:
        self.predictors = predictors
        self.name = name
        self.predictor_input_keys = kwargs.get('predictor_input_keys', None)

    def fit(self, X, y, **kwargs):
        raise NotImplementedError('predictors assumed to be diverse => need to be fitted separately & in advance')

    def predict(self, X):
        if self.predictor_input_keys is None: #inputs are the same for all predictors
            preds = [p.predict(X) for p in self.predictors]
        else:
            preds = [p.predict(X[p_key] if not type(p_key) is list else [X[p_key_i] for p_key_i in p_key])
                     for p, p_key in zip(self.predictors, self.predictor_input_keys)]
        predictions = np.concatenate(preds, axis=1)
        return predictions