import glob
import itertools
import os

import json

import gzip

import argparse

import utils_pareto
from encas.mo_gomea_search import MoGomeaWrapperEnsembleClassification
from encas.random_search import RandomSearchWrapperEnsembleClassification
from greedy_search import GreedySearchWrapperEnsembleClassification

# os.environ['OMP_NUM_THREADS'] = '32'
from os.path import join
from pathlib import Path
import shutil

import yaml
from matplotlib import pyplot as plt

import utils
from after_search.evaluate_stored_outputs import evaluate_stored_whole_experiment_cascade, filter_whole_experiment_cascade
from plot_results.plotting_functions import compare_val_and_test
from utils import set_seed
from utils_pareto import get_best_pareto_up_and_including_iter, get_everything_from_iter

from utils import threshold_gene_to_value_moregranular as threshold_gene_to_value

import numpy as np


def create_problem_data(experiment_paths, max_iters, dataset_type, if_allow_noop, ensemble_ss_names,
                        funs_to_get_subnets_names, if_load_output_distr, dataset_postfix,
                        if_return_logit_gaps=False, if_timm=False):
    cfgs_all, flops_all, iters_all, outputs_all, ss_names_all, experiment_paths_all, logit_gaps_all, original_indices_all = [], [], [], [], [], [], [], []
    alphabet = 0
    assert len(experiment_paths) == len(max_iters)
    if funs_to_get_subnets_names is None:
        funs_to_get_subnets_names = ['cum_pareto'] * len(max_iters)
    elif type(funs_to_get_subnets_names) is not list:
        funs_to_get_subnets_names = [funs_to_get_subnets_names] * len(max_iters)
    name2fun = {'cum_pareto': get_best_pareto_up_and_including_iter, 'last_iter': get_everything_from_iter,
                'all': utils_pareto.get_everything_up_and_including_iter}

    for i, (experiment_path, max_iter, fun_get_subnets_name) in enumerate(zip(experiment_paths, max_iters, funs_to_get_subnets_names)):
        if not if_timm:
            cfgs, _, flops, iters = name2fun[fun_get_subnets_name](experiment_path, max_iter)
        else:
            path_info = os.path.join(utils.NAT_LOGS_PATH, experiment_path, f'output_distrs_{dataset_type}{dataset_postfix}', 'info.json')
            loaded = json.load(open(path_info))
            cfgs = np.array([[x] for x in loaded['net_names']])
            flops = np.array(loaded['flops'])
            iters = np.array([0] * len(cfgs))
        subsample = lambda l: l#[l[0], l[-1]]
        # Note that the subsampling won't work for mo-gomea (at least) because the output_distr is not subsampled, so indexes will refer to wrong outputs
        cfgs, flops, iters = subsample(cfgs), subsample(flops), subsample(iters)

        output_distr_path = os.path.join(utils.NAT_LOGS_PATH, experiment_path, f'output_distrs_{dataset_type}{dataset_postfix}')
        logit_gaps_path = os.path.join(utils.NAT_LOGS_PATH, experiment_path, f'logit_gaps_{dataset_type}{dataset_postfix}')
        if if_load_output_distr:
            assert os.path.isdir(output_distr_path)
            output_distr = []
            n_files = len(glob.glob(os.path.join(output_distr_path, "*.npy.gz")))
            for j in range(n_files):
                print(j)
                with gzip.GzipFile(os.path.join(output_distr_path, f'{j}.npy.gz'), 'r') as f:
                    output_distr.append(np.asarray(np.load(f), dtype=np.float16)[None, ...])
            outputs_all += output_distr

            if if_return_logit_gaps:
                logit_gaps = []
                for j in range(n_files):
                    print(j)
                    with gzip.GzipFile(os.path.join(logit_gaps_path, f'{j}.npy.gz'), 'r') as f:
                        logit_gaps.append(np.asarray(np.load(f), dtype=np.float16)[None, ...])
                logit_gaps_all += logit_gaps

            original_indices_all += list(range(len(output_distr)))
        else:
            outputs_all.append(output_distr_path)
            n_files = len(glob.glob(os.path.join(output_distr_path, '*.npy.gz')))
            original_indices_all += list(range(n_files))

        # if if_return_logit_gaps:
        #     logit_gaps_path = os.path.join('/export/scratch3/aleksand/nsganetv2/logs', experiment_path,
        #                                      f'logit_gaps_cum_pareto_{dataset_type}_logitgaps.npy')
        #     logit_gaps = np.load(logit_gaps_path)
        #     logit_gaps_all.append(logit_gaps)


        cfgs_all += cfgs.tolist()
        flops_all.append(flops.tolist())
        iters_all += iters.tolist()
        alphabet += len(cfgs)
        ss_names_all += [ensemble_ss_names[i]] * len(cfgs)
        experiment_paths_all += [experiment_path] * len(cfgs)

    if if_allow_noop:
        cfgs_all = [[None]] + cfgs_all
        flops_all = [[0]] + flops_all
        iters_all = [iters_all[-1]] + iters_all
        alphabet += 1
        ss_names_all = ['noop'] + ss_names_all
        experiment_paths_all = ['noop'] + experiment_paths_all
        original_indices_all = [None] + original_indices_all

        if if_load_output_distr:
            output_distr_onenet_noop = np.zeros_like(output_distr[0])  # copy arbitrary one to get the shape
            if len(output_distr_onenet_noop.shape) < 3:
                output_distr_onenet_noop = output_distr_onenet_noop[None, ...]
            # output_distr = np.concatenate((output_distr_onenet_noop, output_distr), axis=0)
            outputs_all = [output_distr_onenet_noop] + outputs_all

        if if_return_logit_gaps:
            logit_gaps_onenet_noop = np.zeros_like(logit_gaps_all[0][0])  # copy arbitrary one to get the shape
            if len(logit_gaps_onenet_noop.shape) < 2:
                logit_gaps_onenet_noop = logit_gaps_onenet_noop[None, ...]
            logit_gaps_all = [logit_gaps_onenet_noop] + logit_gaps_all

    if if_load_output_distr:
        preallocated_array = np.zeros((len(outputs_all), outputs_all[0].shape[1], outputs_all[0].shape[2]), dtype=np.float16)
        np.concatenate((outputs_all), axis=0, out=preallocated_array)
        outputs_all = preallocated_array
    if if_return_logit_gaps:
        preallocated_array = np.zeros((len(logit_gaps_all), logit_gaps_all[0].shape[1]), dtype=np.float16)
        np.concatenate((logit_gaps_all), axis=0, out=preallocated_array)
        logit_gaps_all = preallocated_array

    return cfgs_all, flops_all, iters_all, alphabet, outputs_all, ss_names_all, experiment_paths_all, logit_gaps_all, original_indices_all


def create_msunas_config():
    msunas_config_path_starter = os.path.join(nat_logs_path, experiment_paths[0], 'config_msunas.yml')
    msunas_config_path = join(run_path, 'config_msunas.yml')
    shutil.copy(msunas_config_path_starter, msunas_config_path)
    msunas_config = yaml.load(open(msunas_config_path, 'r'), yaml.SafeLoader)
    msunas_config['ensemble_ss_names'] = ensemble_ss_names
    msunas_config['supernet_path'] = supernet_paths
    yaml.dump(msunas_config, open(msunas_config_path, 'w'))
    return msunas_config


def create_pareto(true_errs_out, flops_out, cfgs_out, weight_paths_out, ss_names_out, thresholds_values_out, original_indices_out):
    idx_sort_flops = np.argsort(flops_out)
    true_errs_out = np.array(true_errs_out)[idx_sort_flops]
    flops_out = np.array(flops_out)[idx_sort_flops]
    all_objs = list(zip(true_errs_out, flops_out))
    all_objs = np.array(all_objs)
    pareto_best_cur_idx = utils_pareto.is_pareto_efficient(all_objs)
    true_errs_out = true_errs_out[pareto_best_cur_idx]
    flops_out = flops_out[pareto_best_cur_idx]

    def sort(l):
        return np.array(l)[idx_sort_flops][pareto_best_cur_idx]

    cfgs_out = sort(cfgs_out)
    weight_paths_out = sort(weight_paths_out)
    ss_names_out = sort(ss_names_out)
    thresholds_values_out = sort(thresholds_values_out)
    original_indices_out = sort(original_indices_out)
    return true_errs_out, flops_out, cfgs_out, weight_paths_out, ss_names_out, thresholds_values_out, original_indices_out


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.grid'] = True

    p = argparse.ArgumentParser()
    p.add_argument(f'--config', default='configs_encas/cifar100_DEBUG.yml', type=str)
    p.add_argument(f'--target_run', default=None, type=str)
    parsed_args = vars(p.parse_args())
    encas_config_path = parsed_args['config']
    cfg = yaml.safe_load(open(os.path.join(utils.NAT_PATH, encas_config_path)))

    nat_logs_path, nat_data_path = utils.NAT_LOGS_PATH, utils.NAT_DATA_PATH

    funs_to_get_subnets_names = cfg['funs_to_get_subnets_names']
    dataset_name, dataset_postfix, label_postfix = cfg['dataset'], cfg['dataset_postfix'], cfg['label_postfix']
    SEED = cfg['random_seed']
    n_evals = cfg['n_evals']
    gomea_exe_path = cfg.get('gomea_exe_path', None)

    experiment_names = cfg['input_experiment_names']
    ensemble_size, if_allow_noop = cfg['ensemble_size'], cfg['if_allow_noop']
    max_iters, algo, search_goal = cfg['max_iters'], cfg['algo'], cfg['search_goal']
    dataset_type = cfg['dataset_type']
    target_runs = cfg.get('target_runs', None)
    if parsed_args['target_run'] is not None:
        target_runs = [int(parsed_args['target_run'])]
    ensemble_ss_names, join_or_sep = cfg['input_search_spaces'], 'join' if 'extract' in experiment_names[0].lower() else 'sep'
    out_name_template = cfg['out_name_template']

    cfg['join_or_sep'] = join_or_sep
    cfg['n_inputs'] = len(experiment_names)
    out_name = out_name_template.format(**cfg)
    cfg['out_name'] = out_name

    if algo == 'greedy':
        ensemble_size = 30 # to pad to this size - doesn't influence the algorithm, only the ease of saving
        # define here in order not to screw up out_name

    Path(join(nat_logs_path, out_name)).mkdir(exist_ok=True)
    Path(join(nat_logs_path, out_name, algo)).mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open(join(nat_logs_path, out_name, algo, 'config_encas.yml'), 'w'), default_flow_style=None)

    if_timm = 'timm' in out_name
    if not if_timm:
        supernet_paths = [join(nat_data_path, utils.ss_name_to_supernet_path[ss]) for ss in ensemble_ss_names]
    exp_name_to_get_algo_names = experiment_names[0]

    searcher_class = {'mo-gomea': MoGomeaWrapperEnsembleClassification,
                      'random': RandomSearchWrapperEnsembleClassification,
                      'greedy': GreedySearchWrapperEnsembleClassification}[algo]
    if_load_data = algo == 'greedy'

    labels_path = join(nat_data_path, f'labels_{dataset_name}_{dataset_type}{label_postfix}.npy')
    labels = np.load(labels_path) if if_load_data else labels_path

    for i_algo_folder, f in enumerate(reversed(sorted(os.scandir(join(nat_logs_path, exp_name_to_get_algo_names)), key=lambda e: e.name))):
        if not f.is_dir():
            continue
        nat_algo_name = f.name
        assert i_algo_folder == 0
        # nat_algo_name is whatever method was used in NAT (e.g. "search_algo:mo-gomea!subset_selector:reference"),
        # algo is the algorithm used here (mo-gomea, random, greedy)
        for run_folder in os.scandir(f.path):
            if not run_folder.is_dir():
                continue
            run_idx = run_folder.name
            if target_runs is not None and int(run_idx) not in target_runs:
                print(f'Skipping run {run_idx}')
                continue

            run_path = join(nat_logs_path, out_name, algo, run_idx)
            Path(run_path).mkdir(exist_ok=True)
            log_file_path = os.path.join(run_path, '_log.txt')
            utils.setup_logging(log_file_path)
            experiment_paths = [join(exp, nat_algo_name, run_idx) for exp in experiment_names]
            if not if_timm:
                msunas_config = create_msunas_config()

            cfgs_all, flops_all, iters_all, alphabet, output_distr, ss_names_all, experiment_paths_all, logit_gaps_all, original_indices_all = \
                create_problem_data(experiment_paths, max_iters, dataset_type, if_allow_noop, ensemble_ss_names,
                                    funs_to_get_subnets_names, if_load_data, dataset_postfix,
                                    if_return_logit_gaps=algo == 'greedy', if_timm=if_timm)
            cur_seed = SEED + int(run_idx)
            set_seed(cur_seed)
            flops_all_flattened = list(itertools.chain.from_iterable(flops_all))
            genomes, objs = searcher_class(alphabet, output_distr, flops_all_flattened, labels, if_allow_noop, 
                                           ensemble_size, run_path=run_path, search_goal=search_goal, 
                                           logit_gaps_all=logit_gaps_all, n_evals=n_evals, log_file_path=log_file_path, 
                                           gomea_exe_path=gomea_exe_path).search(cur_seed)

            print(f'{genomes.shape=}')

            plt.plot(objs[:, 1], utils.get_metric_complement(objs[:, 0]), '.', markersize=1)

            cfgs_out, true_errs_out, flops_out, weight_paths_out, ss_names_out, thresholds_values_out, original_indices_out = [], [], [], [], [], [], []
            for i_genome, genome in enumerate(genomes):
                cur_cfg, cur_true_errs, cur_flops, cur_weight_paths, cur_ss_names, thresholds_values, cur_original_indices = [], 0, 0, [], [], [], []
                genome_nets, genome_thresholds = genome[:ensemble_size], genome[ensemble_size:] # thresholds will be empty if not cascade
                for gene in genome_nets:
                    gene = int(gene) # in greedy search the genome is float because half of it are thresholds values
                    cur_cfg.append(cfgs_all[gene][0]) # "[0]" because it's an array of 1 element
                    cur_weight_paths.append(join(nat_logs_path, experiment_paths_all[gene], f'iter_{iters_all[gene]}'))
                    cur_ss_names.append(ss_names_all[gene])
                    cur_original_indices.append(original_indices_all[gene])
                if search_goal == 'cascade':
                    if algo != 'greedy':
                        thresholds_values = [threshold_gene_to_value[x] for x in genome_thresholds]
                    else:
                        #thresholds are not encoded; also, they are logit gaps
                        thresholds_values = genome_thresholds

                cur_true_errs = objs[i_genome, 0]
                cur_flops = objs[i_genome, 1]

                cfgs_out.append(cur_cfg)
                true_errs_out.append(cur_true_errs)
                flops_out.append(cur_flops)
                weight_paths_out.append(cur_weight_paths)
                ss_names_out.append(cur_ss_names)
                thresholds_values_out.append(thresholds_values)
                original_indices_out.append(cur_original_indices)

            true_errs_out, flops_out, cfgs_out, weight_paths_out, ss_names_out, \
            thresholds_values_out, original_indices_out = create_pareto(true_errs_out, flops_out, cfgs_out,
                                                                        weight_paths_out, ss_names_out,
                                                                        thresholds_values_out, original_indices_out)

            plt.plot(flops_out, utils.get_metric_complement(true_errs_out), '-o')
            plt.savefig(join(run_path, 'out.png'), bbox_inches='tight', pad_inches=0) ; plt.show() ; plt.close()

            dict_to_dump = {'true_errs': true_errs_out.tolist(), 'flops': flops_out.tolist(),
                            'cfgs': cfgs_out.tolist(), 'weight_paths': weight_paths_out.tolist(),
                            'search_space_names': ss_names_out.tolist(), 'algo': algo, 'labels_path': labels_path,
                            'dataset_name': dataset_name, 'dataset_type': dataset_type,
                            'original_indices': original_indices_out.tolist(), 'flops_all': flops_all,
                            'ensemble_ss_names': ensemble_ss_names, 'dataset_postfix': dataset_postfix}
            if search_goal == 'cascade':
                dict_to_dump['thresholds'] = thresholds_values_out.tolist()
            yaml.safe_dump(dict_to_dump, open(join(run_path, 'posthoc_ensemble.yml'), 'w'), default_flow_style=None)

    evaluate_stored_whole_experiment_cascade(out_name, 'test', f'labels_{dataset_name}_test.npy', target_algos=[algo], target_runs=target_runs)

    filter_whole_experiment_cascade(out_name, target_algos=[algo], target_runs=target_runs,
        cascade_info_name='posthoc_ensemble_from_stored.yml',
        cascade_info_name_new='posthoc_ensemble_from_stored_filtered.yml')

    compare_val_and_test(out_name, 'test', if_from_stored=True, target_algos=[algo], target_runs=target_runs)