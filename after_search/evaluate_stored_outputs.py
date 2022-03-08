import copy

import json
import numpy as np
import os
import torch
import glob
import gzip

import yaml
from matplotlib import pyplot as plt

import utils
from utils import execute_func_for_all_runs_and_combine

labels_path_prefix = utils.NAT_DATA_PATH

def evaluate_stored_one_run(run_path, dataset_type, path_labels, **kwargs):
    labels = torch.tensor(np.load(os.path.join(labels_path_prefix, path_labels))).cuda() # load, .cuda()
    dataset_postfix = kwargs.get('dataset_postfix', '')
    path_stored_outputs = os.path.join(run_path, f'output_distrs_{dataset_type}{dataset_postfix}')
    max_iter = kwargs.get('max_iter', 15)
    max_iter_path = os.path.join(run_path, f'iter_{max_iter}')
    n_nets = len(glob.glob(os.path.join(path_stored_outputs, '*.npy.gz')))
    info_path = glob.glob(os.path.join(path_stored_outputs, 'info.json'))[0]
    info_path_new = os.path.join(max_iter_path, f'best_pareto_val_and_{dataset_type}{dataset_postfix}.json')

    accs = []
    for i_net in range(n_nets):
        path = os.path.join(path_stored_outputs, f'{i_net}.npy.gz')
        with gzip.GzipFile(path, 'r') as f:
            outs = np.asarray(np.load(f))
        outs = torch.tensor(outs).cuda()
        preds = torch.argmax(outs, axis=-1)
        acc = torch.sum(preds == labels) / len(labels) * 100
        accs.append(acc.item())

    print(accs)
    info = json.load(open(info_path))
    info[dataset_type + dataset_postfix] = accs
    json.dump(info, open(info_path_new, 'w'))
    return accs

def evaluate_stored_whole_experiment(experiment_name, dataset_type, path_labels, **kwargs):
    execute_func_for_all_runs_and_combine(experiment_name, evaluate_stored_one_run, dataset_type=dataset_type,
                                          path_labels=path_labels, **kwargs)

def evaluate_stored_one_run_cascade(run_path, dataset_type, path_labels, **kwargs):
    labels = torch.tensor(np.load(os.path.join(labels_path_prefix, path_labels)))
    cascade_info_name = kwargs.get('cascade_info_name', 'posthoc_ensemble.yml')
    cascade_info_name_new = kwargs.get('cascade_info_name_new', 'posthoc_ensemble_from_stored.yml')
    cascade_info_path = glob.glob(os.path.join(run_path, cascade_info_name))[0]
    info = yaml.safe_load(open(cascade_info_path))
    cfgs, thresholds, original_indices, weight_paths, flops_all, ensemble_ss_names, search_space_names = info['cfgs'], info.get('thresholds', None), info['original_indices'], info['weight_paths'], info['flops_all'], info['ensemble_ss_names'], info['search_space_names']
    if thresholds is None:
        # for ensembles (from ENCAS-ensemble) there are no thresholds, which is equivalent to all thresholds being 1
        # could have a separate method for ENCAS-ensemble, but this fix seems simpler and should lead to the same result
        thresholds = [[1.0] * len(cfgs[0])] * len(cfgs)

    postfix = info.get('dataset_postfix', '')
    if_use_logit_gaps = info['algo'] == 'greedy'
    cascade_info_path_new = cascade_info_path.replace(cascade_info_name, cascade_info_name_new)
    dataset_type_for_path = dataset_type
    labels = labels.cuda()
    outs_cache = {}

    accs = []
    flops_new = []
    for i_cascade, (cfgs_cascade, thresholds_cascade, orig_indices_cascade, weight_paths_cascade, ss_names_cascade) in enumerate(zip(cfgs, thresholds, original_indices, weight_paths, search_space_names)):
        outs = None
        flops_cur = 0
        n_nets_used_in_cascade = 0
        idx_more_predictions_needed = None
        # threshold_idx = 0
        for i_net, (cfg, orig_idx, weight_path, ss_name) in enumerate(zip(cfgs_cascade, orig_indices_cascade, weight_paths_cascade, ss_names_cascade)):
            if cfg is None: #noop
                continue

            # 1. load
            base_path = weight_path[:weight_path.find('/iter')] #need run path
            path = os.path.join(base_path, f'output_distrs_{dataset_type_for_path}{postfix}', f'{orig_idx}.npy.gz')
            if path not in outs_cache:
                with gzip.GzipFile(path, 'r') as f:
                    outs_cur = np.asarray(np.load(f))
                    outs_cur = torch.tensor(outs_cur).cuda()
                    outs_cache[path] = outs_cur
            outs_cur = torch.clone(outs_cache[path])
            if if_use_logit_gaps:
                path = os.path.join(base_path, f'logit_gaps_{dataset_type_for_path}{postfix}', f'{orig_idx}.npy.gz')
                with gzip.GzipFile(path, 'r') as f:
                    logit_gaps_cur = np.asarray(np.load(f))
                logit_gaps_cur = torch.tensor(logit_gaps_cur)

            # 2. predict
            if idx_more_predictions_needed is None:
                idx_more_predictions_needed = torch.ones(outs_cur.shape[0], dtype=torch.bool)
                outs = outs_cur
                n_nets_used_in_cascade = 1
                flops_cur += flops_all[ensemble_ss_names.index(ss_name) + 1][orig_idx] # "+1" because the first one is nooop
                if if_use_logit_gaps:
                    logit_gaps = logit_gaps_cur
            else:
                threshold = thresholds_cascade[i_net - 1]
                if not if_use_logit_gaps:
                    idx_more_predictions_needed[torch.max(outs, dim=1).values > threshold] = False
                else:
                    idx_more_predictions_needed[logit_gaps > threshold] = False

                outs_tmp = outs[idx_more_predictions_needed]  # outs_tmp is needed because I wanna do (in the end) x[idx1][idx2] = smth, and that doesn't modify the original x
                if not if_use_logit_gaps:
                    not_predicted_idx = torch.max(outs_tmp, dim=1).values <= threshold
                else:
                    logit_gap_tmp = logit_gaps[idx_more_predictions_needed]
                    not_predicted_idx = logit_gap_tmp <= threshold
                n_not_predicted = torch.sum(not_predicted_idx).item()
                if n_not_predicted == 0:
                    break

                if not if_use_logit_gaps:
                    n_nets_used_in_cascade += 1
                    coeff1 = (n_nets_used_in_cascade - 1) / n_nets_used_in_cascade # for the current predictions that may already be an average
                    coeff2 = 1 / n_nets_used_in_cascade # for the predictions of the new model

                    outs_tmp[not_predicted_idx] = coeff1 * outs_tmp[not_predicted_idx] \
                                                   + coeff2 * outs_cur[idx_more_predictions_needed][not_predicted_idx]
                    outs[idx_more_predictions_needed] = outs_tmp
                else:
                    # firstly, need to overwrite previous predictions (because they didn't really happen if the gap was too small)
                    outs_tmp[not_predicted_idx] = outs_cur[idx_more_predictions_needed][not_predicted_idx]
                    outs[idx_more_predictions_needed] = outs_tmp

                    # secondly, need to update the logit gap
                    logit_gap_tmp[not_predicted_idx] = logit_gaps_cur[idx_more_predictions_needed][not_predicted_idx]
                    # note that the gap for the previously predicted values will be wrong, but it doesn't matter
                    # because the idx for them has already been set to False
                    logit_gaps[idx_more_predictions_needed] = logit_gap_tmp

                flops_cur += flops_all[ensemble_ss_names.index(ss_name) + 1][orig_idx] * (n_not_predicted / len(labels))

        assert outs is not None
        preds = torch.argmax(outs, axis=-1)
        acc = torch.sum(preds == labels) / len(labels) * 100
        accs.append(acc.item())
        print(f'{i_cascade}: {accs[-1]}')
        flops_new.append(flops_cur)

    print(accs)
    info['val'] = info['true_errs'] # this whole thing is a mess; in plot_results, 'val' is read, but assumed to be true_errs
    info['flops_old'] = info['flops']
    info['flops'] = flops_new
    info[dataset_type] = accs
    yaml.safe_dump(info, open(cascade_info_path_new, 'w'), default_flow_style=None)
    return accs

def evaluate_stored_whole_experiment_cascade(experiment_name, dataset_type, path_labels, **kwargs):
    execute_func_for_all_runs_and_combine(experiment_name, evaluate_stored_one_run_cascade, dataset_type=dataset_type,
                                          path_labels=path_labels, **kwargs)

def filter_one_run_cascade(run_path, cascade_info_name, cascade_info_name_new, **kwargs):
    # filter indices based on val
    # assume that in 'info' the pareto front for 'val' is stored
    # note that even though test was computed for all the cascades for convenience, no selection is done on test.
    cascade_info_path = glob.glob(os.path.join(run_path, cascade_info_name))[0]
    info = yaml.safe_load(open(cascade_info_path))
    cfgs = info['cfgs']

    def create_idx(key_for_filt):
        val_for_filt = info[key_for_filt]
        if_round = True
        if if_round:
            val_for_filt_new = []
            for v in val_for_filt:
                if kwargs.get('subtract_from_100', True):
                    v = 100 - v
                v = round(v * 10) / 10 # wanna have unique digit after comma
                val_for_filt_new.append(v)
            val_for_filt = val_for_filt_new
        cur_best = 0.
        idx_new_pareto = np.zeros(len(cfgs), dtype=bool)
        for i in range(len(cfgs)):
            if val_for_filt[i] > cur_best:
                idx_new_pareto[i] = True
                cur_best = val_for_filt[i]
        return idx_new_pareto

    def filter_by_idx(l, idx):
        return np.array(l)[idx].tolist()

    def filter_info(key_list, idx):
        info_new = copy.deepcopy(info)
        for k in key_list:
            if k in info_new:
                info_new[k] = filter_by_idx(info[k], idx)
        return info_new

    val_key = kwargs.get('val_key', 'val')
    idx_new_pareto = create_idx(val_key)
    info_new = filter_info(['cfgs', 'flops', 'flops_old', 'test', 'thresholds', 'true_errs', 'val', 'weight_paths',
                            'search_space_names', 'original_indices'], idx_new_pareto)
    plt.plot(info['flops'], info['test'], '-o')
    plt.plot(info_new['flops'], info_new['test'], '-o')
    plt.savefig(os.path.join(run_path, 'filtered.png'))
    plt.show()
    plt.close()
    cascade_info_path_new = cascade_info_path.replace(cascade_info_name, cascade_info_name_new)
    yaml.safe_dump(info_new, open(cascade_info_path_new, 'w'), default_flow_style=None)

def filter_whole_experiment_cascade(experiment_name, cascade_info_name, cascade_info_name_new, **kwargs):
    execute_func_for_all_runs_and_combine(experiment_name, filter_one_run_cascade, cascade_info_name=cascade_info_name,
                                          cascade_info_name_new=cascade_info_name_new, **kwargs)