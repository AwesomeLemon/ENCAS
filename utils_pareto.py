import json
import os

import numpy as np

from utils import NAT_LOGS_PATH


def is_pareto_efficient(costs):  # from https://stackoverflow.com/a/40239615/5126900
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def get_best_pareto_from_iter(experiment_path, iter):
    path = os.path.join(NAT_LOGS_PATH, experiment_path)

    obj1_archive = []
    true_errors_archive = []
    configs_archive = []
    with open(os.path.join(path, "iter_{}.stats".format(iter))) as f:
        data = json.load(f)
        for data_archive in data['archive']: # archive always includes candidates
            try:
                (config, perf, flops) = data_archive
            except:
                config, perf, flops, diversity = data_archive
            obj1_archive.append(flops)
            true_errors_archive.append(perf)
            configs_archive.append(config)

    idx_archive_sort_flops = np.argsort(obj1_archive)
    obj1_archive = np.array(obj1_archive)[idx_archive_sort_flops]
    true_errors_archive = np.array(true_errors_archive)[idx_archive_sort_flops]

    all_objs = list(zip(true_errors_archive, obj1_archive))
    all_objs_cur = np.array(all_objs)
    pareto_best_cur_idx = is_pareto_efficient(all_objs_cur)

    return np.array(configs_archive)[idx_archive_sort_flops][pareto_best_cur_idx], \
           true_errors_archive[pareto_best_cur_idx], obj1_archive[pareto_best_cur_idx]


def get_best_pareto_up_and_including_iter(experiment_path, iter):
    path = os.path.join(NAT_LOGS_PATH, experiment_path)

    obj1_archive = []
    true_errors_archive = []
    configs_archive = []
    iterations_archive = [] #need to store with which version of supernet's weights the performance was achieved
    for i in range(iter + 1):
        with open(os.path.join(path, "iter_{}.stats".format(i))) as f:
            data = json.load(f)
            for data_archive in data['archive']: # archive always includes candidates
                try:
                    (config, perf, flops) = data_archive
                except:
                    config, perf, flops, diversity = data_archive
                obj1_archive.append(flops)
                true_errors_archive.append(perf)
                configs_archive.append(config)
                iterations_archive.append(i)

    idx_archive_sort_flops = np.argsort(obj1_archive)
    obj1_archive = np.array(obj1_archive)[idx_archive_sort_flops]
    true_errors_archive = np.array(true_errors_archive)[idx_archive_sort_flops]

    all_objs = list(zip(true_errors_archive, obj1_archive))
    all_objs_cur = np.array(all_objs)
    pareto_best_cur_idx = is_pareto_efficient(all_objs_cur)

    return np.array(configs_archive)[idx_archive_sort_flops][pareto_best_cur_idx], \
           true_errors_archive[pareto_best_cur_idx], \
           obj1_archive[pareto_best_cur_idx], \
           np.array(iterations_archive)[idx_archive_sort_flops][pareto_best_cur_idx]


def get_everything_up_and_including_iter(experiment_path, iter):
    path = os.path.join(NAT_LOGS_PATH, experiment_path)

    obj1_archive = []
    true_errors_archive = []
    configs_archive = []
    iterations_archive = [] #need to store with which version of supernet's weights the performance was achieved
    for i in range(iter + 1):
        with open(os.path.join(path, "iter_{}.stats".format(i))) as f:
            data = json.load(f)
            for data_archive in data['archive']: # archive always includes candidates
                try:
                    (config, perf, flops) = data_archive
                except:
                    config, perf, flops, diversity = data_archive
                obj1_archive.append(flops)
                true_errors_archive.append(perf)
                configs_archive.append(config)
                iterations_archive.append(i)

    idx_archive_sort_flops = np.argsort(obj1_archive)
    obj1_archive = np.array(obj1_archive)[idx_archive_sort_flops]
    true_errors_archive = np.array(true_errors_archive)[idx_archive_sort_flops]

    return np.array(configs_archive)[idx_archive_sort_flops], \
           true_errors_archive, \
           obj1_archive, \
           np.array(iterations_archive)[idx_archive_sort_flops]


def get_everything_from_iter(experiment_path, iter):
    # the only diffs from the fun above are (1) removal of pareto_best_cur_idx (2) returning of iters
    path = os.path.join(NAT_LOGS_PATH, experiment_path)

    obj1_archive = []
    true_errors_archive = []
    configs_archive = []
    with open(os.path.join(path, "iter_{}.stats".format(iter))) as f:
        data = json.load(f)
        for data_archive in data['archive']: # archive always includes candidates
            try:
                (config, perf, flops) = data_archive
            except:
                config, perf, flops, diversity = data_archive
            obj1_archive.append(flops)
            true_errors_archive.append(perf)
            configs_archive.append(config)

    idx_archive_sort_flops = np.argsort(obj1_archive)
    obj1_archive = np.array(obj1_archive)[idx_archive_sort_flops]
    true_errors_archive = np.array(true_errors_archive)[idx_archive_sort_flops]

    return np.array(configs_archive)[idx_archive_sort_flops], \
           true_errors_archive, obj1_archive, np.array([iter] * len(configs_archive))