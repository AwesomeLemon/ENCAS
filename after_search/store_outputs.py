import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
import glob

import utils
from utils import save_gz
from utils_pareto import get_best_pareto_up_and_including_iter
from evaluate import evaluate_many_configs


def store_cumulative_pareto_front_outputs_single_run(experiment_path, dataset_type='test', **kwargs):
    max_iter = kwargs.get('max_iter', 15)
    if_store_float16 = True

    save_path_base = experiment_path
    swa = kwargs.get('swa', None)
    postfix = kwargs.get('postfix', '')
    if swa is not None:
        postfix = f'_swa{swa}' + postfix
    if_store_logit_gaps = kwargs.get('if_store_logit_gaps', True)

    out_folder_path = os.path.join(save_path_base, f'output_distrs_{dataset_type}{postfix}')
    if os.path.exists(out_folder_path) and not kwargs.get('overwrite', True):
        print('Skipping')
        return
    Path(out_folder_path).mkdir(exist_ok=True)
    process_pool = ProcessPoolExecutor(max_workers=1)
    info_dict_path = os.path.join(out_folder_path, f'info.json')
    futures = []
    cnt = 0
    if if_store_logit_gaps:
        out_folder_path_logit_gaps = os.path.join(save_path_base, f'logit_gaps_{dataset_type}{postfix}')
        Path(out_folder_path_logit_gaps).mkdir(exist_ok=True)

    msunas_config_path = os.path.join(utils.NAT_LOGS_PATH, experiment_path, 'config_msunas.yml')
    msunas_config = yaml.safe_load(open(msunas_config_path, 'r'))
    search_space_name = msunas_config.get('search_space', 'ofa')

    n_iters = len(glob.glob(os.path.join(utils.NAT_LOGS_PATH, experiment_path, "iter_*.stats")))
    if_post_hoc_ensemble = 'posthoc' in experiment_path and 'posthocheavy' not in experiment_path
    if n_iters < max_iter and not if_post_hoc_ensemble:
        print(f'Detected an unfinished run (<{max_iter} iterations) => skip')
        return None

    if 'fun_to_get_subnets' in kwargs:
        print('Using fun_to_get_subnets from kwargs!')
        fun_to_get_subnets = kwargs['fun_to_get_subnets']
    else:
        fun_to_get_subnets = get_best_pareto_up_and_including_iter
    cfgs, true_errs, flops, iters = fun_to_get_subnets(experiment_path, max_iter)
    if swa is not None:
        iters = [max_iter] * len(cfgs)
    subsample = lambda l: l# lambda l: l[::3]#[-2:]
    cfgs, true_errs, flops, iters = subsample(cfgs), subsample(true_errs), subsample(flops), subsample(iters)
    print(f'{flops=}')
    cfgs = [list(c) for c in cfgs]  # otherwise it's an ndarray and can't be saved in json
    n_cfgs = len(cfgs)
    fst_same_iter = 0
    last_same_iter = 0
    same_iter = iters[0]
    i = 1
    ensemble_ss_names = None
    if_stored_all = False
    flops_recomputed = []
    accs_test = []
    run_config = None
    while not if_stored_all:
        if (i == n_cfgs) or (i < n_cfgs and iters[i] != same_iter):
            if search_space_name == 'ensemble':
                path_to_supernet_or_its_dir = []
                for supernet_path_cur in msunas_config['supernet_path']:
                    basename = os.path.basename(supernet_path_cur)
                    if swa is not None:
                        basename = utils.transform_supernet_name_swa(basename, swa)
                    path_to_supernet_or_its_dir.append(os.path.join(utils.NAT_LOGS_PATH, experiment_path, f'iter_{same_iter}', basename))
                ensemble_ss_names = msunas_config['ensemble_ss_names']
            else:
                raise NotImplementedError()
            keys_to_return = ['output_distr', 'run_config', 'flops']
            if if_store_logit_gaps:
                keys_to_return.append('logit_gaps')
            accs_test_cur, info = evaluate_many_configs(path_to_supernet_or_its_dir, cfgs[fst_same_iter:last_same_iter+1],
                                            config_msunas=msunas_config, if_test='test' in dataset_type,
                                            search_space_name=search_space_name,
                                            ensemble_ss_names=ensemble_ss_names, info_keys_to_return=keys_to_return, run_config=run_config,
                                            if_use_logit_gaps=False)
            output_distr_per_model = info['output_distr']
            flops_recomputed += info['flops']
            accs_test += accs_test_cur
            if if_store_logit_gaps:
                logit_gaps_per_model = info['logit_gaps']
            if if_store_float16:
                output_distr_per_model = [o.astype(np.float16) for o in output_distr_per_model]
                if if_store_logit_gaps:
                    logit_gaps_per_model = [l.astype(np.float16) for l in logit_gaps_per_model]
            for i_output in range(len(output_distr_per_model)):
                future = process_pool.submit(save_gz, path=os.path.join(out_folder_path, f'{cnt}.npy.gz'),
                                             data=output_distr_per_model[i_output])
                futures.append(future)
                if if_store_logit_gaps:
                    future = process_pool.submit(save_gz, path=os.path.join(out_folder_path_logit_gaps, f'{cnt}.npy.gz'),
                                                 data=logit_gaps_per_model[i_output])
                    futures.append(future)
                cnt += 1
            run_config = info['run_config'][0]  # a hack for speed
            fst_same_iter = i
            last_same_iter = i
            if i < n_cfgs: # to cover the case of the last iter
                same_iter = iters[i]
        else:
            last_same_iter += 1
        i += 1
        if_stored_all = i > n_cfgs

    info_dict = {'val': list(true_errs), dataset_type: list(accs_test), 'flops': list(flops_recomputed),
                 'cfgs': list(cfgs), 'flops_old': list(flops)}
    json.dump(info_dict, open(info_dict_path, 'w'))

    for f in futures:
        f.result()  # wait on everything

def store_cumulative_pareto_front_outputs(experiment_name, dataset_type, **kwargs):
    utils.execute_func_for_all_runs_and_combine(experiment_name, store_cumulative_pareto_front_outputs_single_run,
                                                dataset_type=dataset_type, **kwargs)

if __name__ == '__main__':
    for d in ['val', 'test']:
        store_cumulative_pareto_front_outputs('cifar100_r0_alphaofa_EXTRACT_alpha', d, max_iter=30, swa='20')
        store_cumulative_pareto_front_outputs('cifar100_r0_alphaofa_EXTRACT_ofa12', d, max_iter=30, swa='20')