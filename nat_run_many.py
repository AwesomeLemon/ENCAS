import argparse
import glob
import os
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
import datetime
import torch
from matplotlib import pyplot as plt

import utils
from nat import default_kwargs, main
import yaml
from shutil import copy
import traceback
from concurrent.futures import ThreadPoolExecutor
import cv2

from plot_results.plotting_functions import compare_val_and_test
from after_search.average_weights import swa_for_whole_experiment
from after_search.evaluate_stored_outputs import evaluate_stored_whole_experiment
from after_search.store_outputs import store_cumulative_pareto_front_outputs
from after_search.symlink_imagenet import create_symlinks

cv2.setNumThreads(0)


def create_all_run_kwargs(config_path):
    config_loaded = yaml.safe_load(open(config_path))

    experiment_name = config_loaded['experiment_name']
    print(experiment_name)

    experiment_kwargs = {k: v[0] for k, v in default_kwargs.items()}  # don't need help string
    experiment_kwargs = dict(experiment_kwargs, **config_loaded)
    path_logs = experiment_kwargs['path_logs']  # '/export/scratch3/aleksand/nsganetv2/'
    Path(path_logs).mkdir(exist_ok=True)
    path = os.path.join(path_logs, experiment_name)
    experiment_kwargs['experiment_name'] = experiment_name
    Path(path).mkdir(exist_ok=True)
    copy(config_path, path)

    algo_mods_all = config_loaded['algo_mods_all']
    transform_str = lambda x: x if type(x) != str else x.replace("/", "_").replace(".", "_")
    algo_mods_names = [
        '!'.join(f'{k}:{transform_str(v)}' for k, v in algo_mods.items())
        for algo_mods in algo_mods_all
    ]

    algo_kwargs_all = [dict(experiment_kwargs, **algo_mods) for algo_mods in algo_mods_all]

    seed_offset = experiment_kwargs.get('seed_offset', 0)  # wanna run the 10 runs on different machines
    cur_seed = experiment_kwargs['random_seed'] + seed_offset
    algo_run_kwargs_all = []

    for i_algo, algo_kwargs in enumerate(algo_kwargs_all):
        path_algo = os.path.join(path, algo_mods_names[i_algo])
        Path(path_algo).mkdir(exist_ok=True)
        n_runs = algo_kwargs['n_runs']
        for run in range(n_runs):
            algo_run_kwargs = algo_kwargs.copy()  # because NAT pops the values, which breaks all the runs after the first
            path_algo_run = os.path.join(path_algo, f'{run + seed_offset}')
            algo_run_kwargs['experiment_name'] = path_algo_run
            algo_run_kwargs['random_seed'] = cur_seed
            algo_run_kwargs_all.append(algo_run_kwargs)

            cur_seed += 1

    return experiment_kwargs, algo_run_kwargs_all


def create_config_for_continuation(run_path, target_max_iter):
    config_path = os.path.join(run_path, 'config_msunas.yml')
    config = yaml.safe_load(open(config_path, 'r'))

    exp_name = config['experiment_name']
    n_iters = len(glob.glob(os.path.join(run_path, "iter_*.stats")))
    if n_iters > 0:
        last_iter = n_iters - 1
        if last_iter == target_max_iter:
            return None
        config['resume'] = os.path.join(exp_name, f'iter_{last_iter}.stats')

        supernet_paths = config['supernet_path']
        supernet_paths_new = []
        for p in supernet_paths:
            name = os.path.basename(p)
            supernet_paths_new.append(os.path.join(exp_name, f'iter_{last_iter}', name))
        config['supernet_path'] = supernet_paths_new

    config_path_new = os.path.join(run_path, 'config_msunas_cont.yml')
    yaml.dump(config, open(config_path_new, 'w'))
    return config_path_new


def create_all_run_continue_kwargs(config_path):
    config_loaded = yaml.safe_load(open(config_path))

    experiment_name = config_loaded['experiment_name']
    print(experiment_name)

    experiment_kwargs = {k: v[0] for k, v in default_kwargs.items()}  # don't need help string
    experiment_kwargs = dict(experiment_kwargs, **config_loaded)
    path_logs = experiment_kwargs['path_logs']  # '/export/scratch3/aleksand/nsganetv2/'
    experiment_path = os.path.join(path_logs, experiment_name)
    experiment_kwargs['experiment_name'] = experiment_name

    algo_run_kwargs_all = []

    for f in sorted(os.scandir(experiment_path), key=lambda e: e.name):
        if not f.is_dir():
            continue
        name_cur = f.name
        for run_folder in sorted(os.scandir(f.path), key=lambda e: e.name):
            if not run_folder.is_dir():
                continue
            run_idx = run_folder.name
            run_path = os.path.join(experiment_path, name_cur, run_folder.name)

            config_path_cur = create_config_for_continuation(run_path, experiment_kwargs['iterations'])
            if config_path_cur is not None:
                algo_run_kwargs_all.append(yaml.safe_load(open(config_path_cur, 'r')))

    return experiment_kwargs, algo_run_kwargs_all


def execute_run(algo_run_kwargs):
    try:
        main(algo_run_kwargs)
    except Exception as e:
        print(traceback.format_exc())
        print(e)


def init_worker(zeroeth_gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{zeroeth_gpu}'
    print('cuda = ', os.environ["CUDA_VISIBLE_DEVICES"])


def run_kwargs_many(experiment_kwargs, algo_run_kwargs_all):
    zeroeth_gpu = experiment_kwargs['zeroeth_gpu']
    executor_class = ProcessPoolExecutor
    if experiment_kwargs['if_debug_run']:
        executor_class = ThreadPoolExecutor  # it's easier to debug with threads
    with executor_class(max_workers=experiment_kwargs['n_gpus'], initializer=init_worker,
                        initargs=(zeroeth_gpu,)) as executor:
        print(algo_run_kwargs_all)
        futures = [executor.submit(execute_run, kwargs) for kwargs in algo_run_kwargs_all]
        for f in futures:
            f.result()  # wait on everything

    print(datetime.datetime.now())


def store(store_kwargs):
    print(f'{store_kwargs=}')
    store_cumulative_pareto_front_outputs(store_kwargs['exp_name'], store_kwargs['dataset_type'],
                                          max_iter=store_kwargs['max_iter'], swa=store_kwargs['swa'],
                                          target_runs=store_kwargs['target_runs'])

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    p = argparse.ArgumentParser()
    p.add_argument(f'--config', default='configs_nat/cifar100_q0_ofa10_sep_DEBUG.yml', type=str)
    p.add_argument(f'--continue', default=False, action='store_true')
    cfgs = vars(p.parse_args())

    config_path = cfgs['config']

    # 1. run NAT
    if not cfgs['continue']:
        experiment_kwargs, algo_run_kwargs_all = create_all_run_kwargs(config_path)
    else:
        experiment_kwargs, algo_run_kwargs_all = create_all_run_continue_kwargs(config_path)
    run_kwargs_many(experiment_kwargs, algo_run_kwargs_all)

    # 2 (optional). do SWA, store subnetwork outputs, compare validation & test
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.grid'] = True
    exp_name = experiment_kwargs['experiment_name']
    max_iter = experiment_kwargs['iterations']
    if_store = experiment_kwargs['if_store']
    dataset = experiment_kwargs['dataset']
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{experiment_kwargs["zeroeth_gpu"]}'
    swa = experiment_kwargs.get('post_swa', None)
    if len(algo_run_kwargs_all) > 0:  # == 0 can occur with 'continue'
        target_runs = [int(x['experiment_name'][-1]) for x in algo_run_kwargs_all]
    else:
        target_runs = list(range(experiment_kwargs['seed_offset'],
                                 experiment_kwargs['seed_offset'] + experiment_kwargs['n_runs']))

    if dataset == 'imagenet':
        # for imagenet weights are not trained => stored only once, but my code needs a supernet-per-metaiteration
        # => symlink
        utils.execute_func_for_all_runs_and_combine(exp_name, create_symlinks, target_runs=target_runs)

    if swa is not None:
        for supernet in experiment_kwargs['supernet_path']:
            swa_for_whole_experiment(exp_name, range(max_iter + 1 - swa, max_iter + 1),
                                     os.path.basename(supernet), target_runs=target_runs)
        if not if_store:
            compare_val_and_test(exp_name, f'test_swa{swa}', swa=swa, max_iter=max_iter, target_runs=target_runs)

    if if_store:
        zeroeth_gpu = experiment_kwargs['zeroeth_gpu']


        kwargs_for_store = [
            dict(exp_name=exp_name, dataset_type='val', max_iter=max_iter, swa=swa, target_runs=target_runs),
            dict(exp_name=exp_name, dataset_type='test', max_iter=max_iter, swa=swa, target_runs=target_runs)
        ]

        n_workers = 1
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker,
                            initargs=(zeroeth_gpu,)) as executor:
            futures = [executor.submit(store, kwargs) for kwargs in kwargs_for_store]
            for f in futures:
                f.result()  # wait on everything

        test_name = 'test' if swa is None else f'test_swa{swa}'
        dataset_to_label_path = {'cifar100': 'labels_cifar100_test.npy', 'cifar10': 'labels_cifar10_test.npy',
                                 'imagenet': 'labels_imagenet_test.npy'}
        evaluate_stored_whole_experiment(exp_name, test_name, dataset_to_label_path[dataset],
                                         max_iter=max_iter, target_runs=target_runs)
        compare_val_and_test(exp_name, test_name, max_iter=max_iter, target_runs=target_runs)
    else:
        if swa is None:
            compare_val_and_test(exp_name, f'test', max_iter=max_iter, target_runs=target_runs)
