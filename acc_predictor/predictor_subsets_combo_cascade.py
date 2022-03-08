import numpy as np


class PredictorSubsetsComboCascade:
    '''
    Contains several base predictors, with each operating on a subset of the input.
    A meta-predictor combines their outputs.
    '''
    def __init__(self, predictor_class, predictor_final, input_sizes, alphabet, alphabet_lb, **kwargs) -> None:
        self.n_predictors = len(input_sizes)
        self.input_sizes = input_sizes
        self.predictors = []
        self.predictor_final = predictor_final
        input_sizes_cumsum = np.cumsum(input_sizes)
        self.input_ranges = list(zip([0] + list(input_sizes_cumsum[:-1]), input_sizes_cumsum))
        alphabets = [alphabet[s:e] for s, e in self.input_ranges]
        alphabets_lb = [alphabet_lb[s:e] for s, e in self.input_ranges]
        for input_size, a, a_lb in zip(input_sizes, alphabets, alphabets_lb):
            p = predictor_class(alphabet=a, alphabet_lb=a_lb, **kwargs)
            self.predictors.append(p)

        self.name = 'rbf_ensemble_per_ensemble_member_combo_cascade'

    def fit(self, X, targets, **kwargs):
        targets_sep, targets_ens = targets['metrics_sep'], targets['metrics_ens']
        # assume both X and y are lists with the same lengths as the number of predictors
        for i, p in enumerate(self.predictors):
            p.fit(X[i], targets_sep[i], **kwargs)
        targets_sep_stacked = np.stack(targets_sep, axis=1)

        positions_and_thresholds = kwargs['inputs_additional']['inputs_for_flops'][:, :-self.n_predictors]
        acc_sep_and_pos_and_thr = np.hstack((targets_sep_stacked, positions_and_thresholds))
        self.predictor_final.fit(acc_sep_and_pos_and_thr, targets_ens)

    def predict(self, X):
        X_for_acc, X_for_flops = X
        X_sep = [X_for_acc[:, s:e] for s, e in self.input_ranges]
        out = []
        for x, p in zip(X_sep, self.predictors):
            preds = p.predict(x)
            out.append(preds)
        out = np.hstack(out)
        pos_and_thr = X_for_flops[:, :-self.n_predictors] # n_samples, N positions + N thresholds
        acc_sep_and_pos_and_thr = np.hstack((out, pos_and_thr))
        res = self.predictor_final.predict(acc_sep_and_pos_and_thr)
        return res