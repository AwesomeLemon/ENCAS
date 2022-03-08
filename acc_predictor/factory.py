import numpy as np

from acc_predictor.predictor_container import PredictorContainer
from acc_predictor.predictor_subsets import PredictorSubsets
from acc_predictor.predictor_subsets_combo_cascade import PredictorSubsetsComboCascade
from acc_predictor.rbf import RBF
from acc_predictor.rbf_ensemble import RBFEnsemble


def get_acc_predictor(model, inputs, targets, alphabet, alphabet_lb, **kwargs):
    ensemble_size = kwargs.get('ensemble_size', 500)

    if model == 'rbf':
        predictor = RBF(alphabet=alphabet, alphabet_lb=alphabet_lb)
        predictor.fit(inputs, targets)

    elif model == 'rbf_ensemble':
        predictor = RBFEnsemble(ensemble_size=ensemble_size, alphabet=alphabet, alphabet_lb=alphabet_lb)
        predictor.fit(inputs, targets)

    elif model == 'rbf_ensemble_per_ensemble_member_cascade': # need to predict flops
        input_sizes = [x.shape[1] for x in inputs]

        acc_predictor = PredictorSubsets(RBFEnsemble, input_sizes, alphabet, alphabet_lb, ensemble_size=ensemble_size)
        acc_predictor.fit(inputs, targets['metrics_sep'])

        flops_predictor = RBFEnsemble(ensemble_size=ensemble_size,
                                     alphabet=kwargs['inputs_additional']['inputs_for_flops_alphabet'],
                                     alphabet_lb=kwargs['inputs_additional']['inputs_for_flops_alphabet_lb'])
        flops_predictor.fit(kwargs['inputs_additional']['inputs_for_flops'], targets['flops_cascade'])

        predictor = PredictorContainer([acc_predictor, flops_predictor], 'rbf_ensemble_per_ensemble_member_cascade',
                                           predictor_input_keys=['for_acc', 'for_flops'])

    elif model == 'rbf_ensemble_per_ensemble_member_cascade_combo': # need to predict flops; predict acc not by averaging but by another predictor
        input_sizes = [x.shape[1] for x in inputs]
        n_supernets = len(input_sizes)
        flops_alphabet = kwargs['inputs_additional']['inputs_for_flops_alphabet']
        flops_alphabet_lb = kwargs['inputs_additional']['inputs_for_flops_alphabet_lb']

        # this predictor takes N predicted errors, N positions, N thresholds
        # I wanna create the alphabets & stuff from what we already have, i.e. alphabets for flops,
        # they contain N positions, then N thresholds, then N flops, ergo:
        alphabet_pos_and_thr = flops_alphabet[:-n_supernets]
        alphabet_lb_pos_and_thr = flops_alphabet_lb[:-n_supernets]
        combo_alphabet = np.concatenate([np.array([100] * n_supernets), alphabet_pos_and_thr])
        combo_alphabet_lb = np.concatenate([np.array([0] * n_supernets), alphabet_lb_pos_and_thr])
        acc_predictor_combo = RBFEnsemble(ensemble_size=ensemble_size, alphabet=combo_alphabet, alphabet_lb=combo_alphabet_lb)

        acc_predictor = PredictorSubsetsComboCascade(RBFEnsemble, acc_predictor_combo,
                                                     input_sizes, alphabet, alphabet_lb, ensemble_size=ensemble_size)
        acc_predictor.fit(inputs, targets, inputs_additional=kwargs['inputs_additional'])

        flops_predictor = RBFEnsemble(ensemble_size=ensemble_size, alphabet=flops_alphabet, alphabet_lb=flops_alphabet_lb)
        flops_predictor.fit(kwargs['inputs_additional']['inputs_for_flops'], targets['flops_cascade'])

        predictor = PredictorContainer([acc_predictor, flops_predictor], 'rbf_ensemble_per_ensemble_member_cascade_combo',
                                           predictor_input_keys=[['for_acc', 'for_flops'], 'for_flops'])

    else:
        raise NotImplementedError

    return predictor

