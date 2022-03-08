import re

import os
import json
from collections import defaultdict

import glob
from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt
import itertools
from textwrap import fill
from PIL import Image
import yaml
import hashlib
from pdf2image import convert_from_path

import utils
from utils_pareto import is_pareto_efficient, get_best_pareto_from_iter, get_best_pareto_up_and_including_iter
from scipy.stats import spearmanr

from evaluate import evaluate_many_configs


nsga_logs_path = utils.NAT_LOGS_PATH

from utils import images_list_to_grid_image

def eval_cumulative_pareto_front_single_run(experiment_path, dataset_type='test', **kwargs):
    max_iter = kwargs.get('max_iter', 15)
    max_iter_path = os.path.join(nsga_logs_path, experiment_path, f'iter_{max_iter}')

    if_post_hoc_ensemble = 'posthoc' in experiment_path and 'posthocheavy' not in experiment_path
    save_path_base = experiment_path if if_post_hoc_ensemble else max_iter_path
    dataset_postfix = kwargs.get('dataset_postfix', '')
    swa = kwargs.get('swa', None)

    if if_post_hoc_ensemble:
        ensemble_information_path = os.path.join(save_path_base, f'posthoc_ensemble_from_stored_filtered.yml')
        ensemble_information = yaml.safe_load(open(ensemble_information_path, 'r'))
        dataset_postfix = ensemble_information['dataset_postfix']
        if 'swa' in dataset_postfix:
            swa = int(re.search('(\d+)(?!.*\d)', dataset_postfix)[0])  # last number in the name


    save_output_path = os.path.join(save_path_base, f'best_pareto_val_and_{dataset_type}{dataset_postfix}.json')

    if_from_stored_posthoc = kwargs.get('if_from_stored', False)
    if if_from_stored_posthoc:
        save_output_path = os.path.join(save_path_base, f'posthoc_ensemble_from_stored_filtered.yml')
        if not os.path.exists(save_output_path):
            save_output_path = os.path.join(save_path_base, f'posthoc_ensemble_from_stored.yml')

    if not (if_from_stored_posthoc and Path(save_output_path).is_file()):
        # need this both when restoring & producing the stuff in the not-store scenario
        nat_config_path = os.path.join(nsga_logs_path, experiment_path, 'config_msunas.yml')
        nat_config = yaml.safe_load(open(nat_config_path, 'r'))

        search_space_name = nat_config.get('search_space', 'ofa')

        n_iters = len(glob.glob(os.path.join(nsga_logs_path, experiment_path, "iter_*.stats")))
        if n_iters < max_iter and not if_post_hoc_ensemble:
            print(f'Detected an unfinished run (<{max_iter} iterations) => skip')
            return None

    if Path(save_output_path).is_file():
        if not if_from_stored_posthoc:
            loaded_data = json.load(open(save_output_path))
        else:
            loaded_data = yaml.safe_load(open(save_output_path))
            nat_config = {'dataset': loaded_data['dataset_name']}
        val = loaded_data['val']
        test = loaded_data[dataset_type]
        flops = loaded_data['flops']
        cfgs = loaded_data['cfgs']
        print(cfgs[-1])

        # true_errs = 100 - np.array(val)
        # accs_test = 100 - np.array(test)
        true_errs = np.array(val)
        accs_test = np.array(test)

        print(accs_test[-1])
    else:
        accs_test = []
        run_config = None
        if not if_post_hoc_ensemble:
            if search_space_name == 'reproduce_nat':
                cfgs, true_errs, flops = get_best_pareto_from_iter(experiment_path, max_iter)
                iters = [max_iter] * len(cfgs)
            else:
                cfgs, true_errs, flops, iters = get_best_pareto_up_and_including_iter(experiment_path, max_iter)
                if swa is not None:
                    iters = [max_iter] * len(cfgs)
            subsample = lambda l: l#[-1:]# lambda l: l[::3]#[-2:]
            cfgs, true_errs, flops, iters = subsample(cfgs), subsample(true_errs), subsample(flops), subsample(iters)
            print(f'{flops=}')
            if search_space_name == 'ensemble':
                cfgs = [list(c) for c in cfgs]  # otherwise it's an ndarray and can't be saved in json
            n_cfgs = len(cfgs)
            fst_same_iter = 0
            last_same_iter = 0
            same_iter = iters[0]
            i = 1
            if_predicted_all = False
            flops_recomputed = []
            while not if_predicted_all:
                if (i == n_cfgs) or (i < n_cfgs and iters[i] != same_iter):
                    path_to_supernet_or_its_dir = []
                    for supernet_path_cur in nat_config['supernet_path']:
                        basename = os.path.basename(supernet_path_cur)
                        if swa is not None:
                            basename = utils.transform_supernet_name_swa(basename, swa)
                        path_to_supernet_or_its_dir.append(os.path.join(nsga_logs_path, experiment_path, f'iter_{same_iter}', basename))
                    ensemble_ss_names = nat_config['ensemble_ss_names']
                    accs_test_cur, info = evaluate_many_configs(path_to_supernet_or_its_dir, cfgs[fst_same_iter:last_same_iter+1],
                                                          config_msunas=nat_config,
                                                          if_test='test' in dataset_type, search_space_name=search_space_name,
                                                          ensemble_ss_names=ensemble_ss_names,
                                                          info_keys_to_return=['flops', 'run_config'], run_config=run_config,
                                                                if_use_logit_gaps=False)
                    accs_test += accs_test_cur
                    flops_recomputed += info['flops']
                    run_config = info['run_config'][0] # a hack for speed

                    fst_same_iter = i
                    last_same_iter = i
                    if i < n_cfgs: # to cover the case of the last iter
                        same_iter = iters[i]
                else:
                    last_same_iter += 1
                i += 1
                if_predicted_all = i > n_cfgs
            flops = flops_recomputed
        else:
            # loaded = yaml.safe_load(open(os.path.join(experiment_path, 'posthoc_ensemble.yml'), 'r'))
            loaded = yaml.safe_load(open(os.path.join(experiment_path, 'posthoc_ensemble_from_stored_filtered.yml'), 'r')) # need filtered => require from_stored
            cfgs, true_errs, flops, weight_paths,  = loaded['cfgs'], loaded['true_errs'], loaded['flops'], loaded['weight_paths']
            subsample = lambda l: l#[-1:]#[::20] + l[-2:] #[::10] #[::3]#
            cfgs, true_errs, flops, weight_paths = subsample(cfgs), subsample(true_errs), subsample(flops), subsample(weight_paths)
            # ensemble_ss_names = nat_config['ensemble_ss_names']
            search_space_names = subsample(loaded['search_space_names'])
            if_cascade = 'thresholds' in loaded
            if if_cascade:
                thresholds = subsample(loaded['thresholds'])
            algo = loaded.get('algo', None)
            print(f'{flops=}')
            flops_recomputed = []
            for i, (cfg, weight_path_cur_ensemble) in enumerate(zip(cfgs, weight_paths)):
                path_to_supernet_or_its_dir = []
                ensemble_ss_names = search_space_names[i]
                ss_name_to_supernet_path = {'ofa12': 'supernet_w1.2', 'ofa10': 'supernet_w1.0',
                                            'alphanet': 'alphanet_pretrained.pth.tar',
                                            'attn': 'attentive_nas_pretrained.pth.tar',
                                            'proxyless': 'ofa_proxyless_d234_e346_k357_w1.3',
                                            'noop': 'noop',}
                supernet_paths = [ss_name_to_supernet_path[ss_name] for ss_name in ensemble_ss_names]

                ss_name_to_expected_ss_name = {'ofa12': 'ofa', 'ofa10': 'ofa', 'ofa': 'ofa',
                                               'alphanet': 'alphanet', 'attn': 'alphanet',
                                               'proxyless': 'proxyless',
                                               'noop': 'noop'}
                ensemble_ss_names = [ss_name_to_expected_ss_name[ss] for ss in ensemble_ss_names]

                for supernet_path_from_config, weights_to_use_path in zip(supernet_paths, weight_path_cur_ensemble):
                    basename = os.path.basename(supernet_path_from_config)
                    if swa is not None:
                        basename = utils.transform_supernet_name_swa(basename, swa)
                        weights_to_use_path = re.sub(r'iter_\d+', f'iter_{max_iter}', weights_to_use_path)
                    path_to_supernet_or_its_dir.append(os.path.join(weights_to_use_path, basename))

                if if_cascade:
                    thresholds_cur = thresholds[i]
                    if algo is not None and algo == 'greedy':
                        # remove trailing zeros
                        thresholds_cur = [t for t in thresholds_cur if t != 0]

                accs_test_cur, info = evaluate_many_configs(path_to_supernet_or_its_dir, [cfg],
                             config_msunas=nat_config, if_test='test' in dataset_type,
                             search_space_name=search_space_name, ensemble_ss_names=ensemble_ss_names,
                             info_keys_to_return=['flops', 'run_config'], run_config=run_config,
                             thresholds=None if not if_cascade else thresholds_cur,
                             if_use_logit_gaps=algo is not None and algo == 'greedy')
                accs_test += accs_test_cur
                flops_recomputed += info['flops']
                run_config = info['run_config'][0]  # a hack for speed
            flops = flops_recomputed

        print(accs_test)
        print(f'{type(true_errs)}, {type(accs_test)}, {type(flops)}, {type(cfgs)}')
        print(f'{true_errs=}')
        print(f'{flops=}')
        print(f'{cfgs=}')
        dict_to_dump = {'val': list(true_errs), dataset_type: list(accs_test), 'flops': list(flops), 'cfgs': list(cfgs)}
        with open(save_output_path, 'w') as handle:
            json.dump(dict_to_dump, handle)
    accs_val = utils.get_metric_complement(np.array(true_errs))
    plt.plot(flops, accs_val, '-o', label='val')
    plt.plot(flops, accs_test, '-o', label=dataset_type)
    plt.legend()
    plt.title(fill(experiment_path + f'; corr={spearmanr(accs_val, accs_test)[0]:.2f}', 70))
    plt.xlabel('Flops')
    plt.ylabel('Accuracy')
    if 'cifar100' in experiment_path:
        if np.median(accs_test) > 70:
            plt.xlim(0, 3700)
            plt.ylim(70, 90)
        # pass
    elif 'cifar10' in experiment_path:
        if np.median(accs_test) > 85:
            plt.xlim(0, 3700)
            plt.ylim(85, 100)

    plt_path = os.path.join(save_path_base, f'best_pareto_val_and_{dataset_type}.png')
    plt.savefig(plt_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    return plt_path



def compare_test_many_experiments(experiment_names, dataset_types, max_iters=None, annotation_shifts=None,
                                  if_plot_many_in_one=False, algo_names=None, target_runs=None, **kwargs):
    if max_iters is None:
        max_iters = [15] * len(experiment_names)
    elif type(max_iters) is int:
        max_iters = [max_iters] * len(experiment_names)
    if annotation_shifts is None:
        annotation_shifts = [(-10, 10 * (-1) ** (i + 1)) for i in range(len(experiment_names))]
    else:
        annotation_shifts = [(-10, 10 * sh) for sh in annotation_shifts]
    if not (type(dataset_types) is list):
        dataset_types = [dataset_types] * len(experiment_names)
    nsga_path = utils.NAT_PATH
    nsga_logs_path = utils.NAT_LOGS_PATH
    tmp_path = os.path.join(nsga_path, '.tmp')
    experiment_and_datatype_to_seed_to_obj0 = defaultdict(dict)
    experiment_and_datatype_to_seed_to_obj1 = defaultdict(dict)
    n_seeds = []
    # 1. read data
    for i_exp, (experiment_name, max_iter, dataset_type) in enumerate(zip(experiment_names, max_iters, dataset_types)):
        postfix = '' if algo_names is None else algo_names[i_exp]
        if 'logs_classification' not in experiment_name:
            experiment_path = os.path.join(nsga_logs_path, experiment_name)
            # load values // assume they are already computed
            for f in reversed(sorted(os.scandir(experiment_path), key=lambda e: e.name)):
                if not f.is_dir():
                    continue
                name_cur = f.name
                if algo_names is not None and algo_names[i_exp] != name_cur:
                    continue
                n_seeds_cur = 0
                for run_folder in sorted(os.scandir(f.path), key=lambda e: e.name):
                    if not run_folder.is_dir():
                        continue
                    run_idx = int(run_folder.name)
                    if target_runs is not None and run_idx not in target_runs:
                        continue
                    run_path = os.path.join(experiment_path, name_cur, str(run_idx))

                    if 'posthoc' in experiment_name:
                        stored_accs_path = os.path.join(run_path, f'best_pareto_val_and_{dataset_type}.json')
                    else:
                        stored_accs_path = os.path.join(run_path, f'iter_{max_iter}', f'best_pareto_val_and_{dataset_type}.json')

                    if os.path.exists(stored_accs_path):
                        loaded_data = json.load(open(stored_accs_path))
                        test = np.array(loaded_data[dataset_type])
                        flops = np.array(loaded_data['flops'])
                    else:
                        # maybe it's posthoc + from_stored?
                        # stored_accs_path = os.path.join(run_path, 'posthoc_ensemble_from_stored.yml')
                        stored_accs_path = os.path.join(run_path, f'posthoc_ensemble_from_stored_filtered.yml')
                        if not os.path.exists(stored_accs_path):
                            stored_accs_path = os.path.join(run_path, f'posthoc_ensemble_from_stored.yml')
                        loaded_data = yaml.safe_load(open(stored_accs_path))
                        test = np.array(loaded_data[dataset_type])
                        flops = np.array(loaded_data['flops'])

                    experiment_and_datatype_to_seed_to_obj0[experiment_name + '_' + dataset_type + postfix][run_idx] = test
                    experiment_and_datatype_to_seed_to_obj1[experiment_name + '_' + dataset_type + postfix][run_idx] = flops
                    n_seeds_cur += 1
                n_seeds.append(n_seeds_cur)
        else:
            experiment_path = os.path.join(nsga_path, experiment_name)
            n_seeds_cur = 0
            for run_folder in sorted(os.scandir(experiment_path), key=lambda e: e.name):
                if not run_folder.is_dir():
                    continue
                run_idx = int(run_folder.name)
                if target_runs is not None and run_idx not in target_runs:
                    continue
                run_path = os.path.join(experiment_path, str(run_idx))

                data = yaml.safe_load(open(os.path.join(run_path, 'data.yml')))
                test = data['test']
                flops = data['flops']
                experiment_and_datatype_to_seed_to_obj0[experiment_name + '_' + dataset_type + postfix][run_idx] = test
                experiment_and_datatype_to_seed_to_obj1[experiment_name + '_' + dataset_type + postfix][run_idx] = flops
                n_seeds_cur += 1
            n_seeds.append(n_seeds_cur)


    n_seeds = min(n_seeds)
    image_paths = []

    map_exp_names = {'cifar10_r0_proxyless_sep': 'NAT + ProxylessNAS',
                     'cifar10_r0_ofa10_sep': 'NAT + OFA-w1.0',
                     'cifar10_r0_ofa12_sep': 'NAT + OFA-w1.2',
                     'cifar10_r0_attn_sep': 'NAT + AttentiveNAS',
                     'cifar10_r0_alpha_sep': 'NAT + AlphaNet',
                     'cifar10_reproducenat': 'NAT (reproduced)',

                     'cifar100_r0_proxyless_sep': 'NAT + ProxylessNAS',
                     'cifar100_r0_ofa10_sep': 'NAT + OFA-w1.0',
                     'cifar100_r0_ofa12_sep': 'NAT + OFA-w1.2',
                     'cifar100_r0_attn_sep': 'NAT + AttentiveNAS',
                     'cifar100_r0_alpha_sep': 'NAT + AlphaNet',
                     'cifar100_reproducenat': 'NAT (reproduced)',

                     'imagenet_r0_proxyless_sep': 'NAT + ProxylessNAS',
                     'imagenet_r0_ofa10_sep': 'NAT + OFA-w1.0',
                     'imagenet_r0_ofa12_sep': 'NAT + OFA-w1.2',
                     'imagenet_r0_attn_sep': 'NAT + AttentiveNAS',
                     'imagenet_r0_alpha_sep': 'NAT + AlphaNet',
                     }
    # map_exp_names = {}
    # experiment_names_pretty = [map_exp_names.get(name, name).replace('+', '\n+') for name in experiment_names]
    experiment_names_pretty = [map_exp_names.get(name, name) for name in experiment_names]

    def set_lims(exp_name):
        if 'cifar100' in exp_name:
            pass
            # plt.xlim(0, 3800)
            # plt.xlim(0, 2750)
            # plt.xlim(0, 2200)
            # plt.ylim(75, 90)
            # plt.xscale('log')
        elif 'cifar10' in exp_name:
            pass
            # plt.xlim(0, 2750)
            # plt.xlim(0, 3700)
            # plt.xlim(0, 2200)
            # plt.ylim(95, 99)
        elif 'imagenet' in exp_name:
            pass
            # plt.xlim(100, 2100)
            # plt.ylim(77, 83)
            # plt.xlim(left=200, right=2800)
            # plt.xscale('log')
        plt.ylabel('Accuracy')
        if kwargs.get('if_log_scale_x', False):
            plt.xscale('log')
            plt.xlabel('Avg. MFLOPS - log scale')

    markers = ['-o', '-X', '-+', '-_']
    if_add_dataset_type_to_label = True
    if_show_title = True
    if_save_as_pdf = kwargs.get('pdf', False)
    if not if_plot_many_in_one: # either make separate plots (this if-branch), or plot shaded area (the other branch)
        for seed in range(n_seeds):
            plt.figure(figsize=(10, 8))
            cur_marker_idx = 0

            for i, (experiment_name, dataset_type) in enumerate(zip(experiment_names, dataset_types)):
                postfix = '' if algo_names is None else algo_names[i]
                obj0 = experiment_and_datatype_to_seed_to_obj0[experiment_name + '_' + dataset_type + postfix][seed]
                obj1 = experiment_and_datatype_to_seed_to_obj1[experiment_name + '_' + dataset_type + postfix][seed]
                name_postfix = (f' ({dataset_type})' + postfix) if if_add_dataset_type_to_label else ''
                plt.plot(obj1, obj0, markers[cur_marker_idx], label=experiment_names_pretty[i] + name_postfix)#, alpha=0.7)
                # plt.annotate(r"$\bf{" + f'{obj0[-1]:.2f}' + '}$', xy=(obj1[-1], obj0[-1]), xytext=annotation_shifts[i],
                #                 textcoords='offset points')
                cur_marker_idx = (cur_marker_idx + 1) % len(markers)

            set_lims(experiment_names[0])

            plt.xlabel('FLOPS')
            if if_show_title:
                plt.title(f'{dataset_types}, {seed=}')
            # plt.title(f'Performance on {dataset_type}')
            plt.subplots_adjust(bottom=0.3)
            plt.legend(bbox_to_anchor=(-0.1, -0.1), mode="expand")
            # plt.legend(bbox_to_anchor=(1.12, -0.15), ncol=2)
            im_path = os.path.join(tmp_path, f'{seed}.png')
            plt.savefig(im_path, bbox_inches='tight', pad_inches=0)
            image_paths.append(im_path)
            plt.show()
    else:
        markers = ['-o', '-s', '-X', '-+', '-v', '-^', '-<', '->', '-D']
        cur_marker_idx = 0
        plt.figure(figsize=(6.6, 6.6))
        for i_exp, (experiment_name, dataset_type) in enumerate(zip(experiment_names, dataset_types)):
            postfix = '' if algo_names is None else algo_names[i_exp]
            obj0_all, obj1_all, test_hv_all = [], [], []
            for seed in range(n_seeds): # reversed because wanna plot 0-th seed 5 lines below
                obj0 = np.array(experiment_and_datatype_to_seed_to_obj0[experiment_name + '_' + dataset_type + postfix][seed])
                obj1 = np.array(experiment_and_datatype_to_seed_to_obj1[experiment_name + '_' + dataset_type + postfix][seed])

                obj0_all.append(obj0)
                obj1_all.append(obj1)

                # compute test hypervolume
                worst_top1_err, worst_flops = 40, 4000
                ref_pt = np.array([worst_top1_err, worst_flops])
                test_hv = utils.compute_hypervolume(ref_pt, np.column_stack([100 - obj0, obj1]), if_increase_ref_pt=False)
                test_hv_all.append(test_hv)

            idx_median = np.argsort(test_hv_all)[len(test_hv_all) // 2]
            print(f'{idx_median=}')
            legend_labels = kwargs.get('legend_labels', [postfix] * len(experiment_names))
            postfix = legend_labels[i_exp]
            # print(f'{obj1_all[idx_median]=}')
            plt.plot(obj1_all[idx_median], obj0_all[idx_median], markers[cur_marker_idx], label=experiment_names_pretty[i_exp] if postfix == '' else postfix)
            cur_marker_idx = (cur_marker_idx + 1) % len(markers)

            if 'logs_classification' not in experiment_name:
                obj0_all = list(itertools.chain(*obj0_all))
                obj1_all = list(itertools.chain(*obj1_all))
                idx_sort_flops = np.argsort(obj1_all)
                obj0_all = np.array(obj0_all)[idx_sort_flops]
                obj1_all = np.array(obj1_all)[idx_sort_flops]

                objs_all_for_pareto = np.vstack((100 - obj0_all, obj1_all)).T
                idx_pareto = is_pareto_efficient(objs_all_for_pareto)
                pareto = objs_all_for_pareto[idx_pareto].T

                # by "antipareto" I mean the bottom edge of the point set; in this terminology, all the points lie above antipareto and below pareto
                objs_all_for_antipareto = np.vstack((obj0_all, -obj1_all)).T
                idx_antipareto = is_pareto_efficient(objs_all_for_antipareto)
                antipareto = objs_all_for_antipareto[idx_antipareto].T

                fill_x = np.append(pareto[1], -antipareto[1][::-1])
                fill_y = np.append(100 - pareto[0], antipareto[0][::-1])

                plt.fill(fill_x, fill_y, alpha=0.5)

        plt.xlabel('Avg. MFLOPS')
        set_lims(experiment_names[0])
        plt.legend()
        name = kwargs.get('out_name', f'{str(experiment_names).replace(r"/", "_")[:100]}')
        im_path = os.path.join(tmp_path, name + ('.pdf' if if_save_as_pdf else '.png'))
        plt.savefig(im_path, bbox_inches='tight', pad_inches=0.01)#, dpi=300)
        image_paths.append(im_path)
        plt.show()

    if not if_save_as_pdf:
        w, h = Image.open(image_paths[0]).size
        open_or_create_image = lambda path: Image.new(mode='RGB', size=(w, h)) if path is None else Image.open(path)
    else:
        open_or_create_image = lambda path: convert_from_path(path)[0]
    ims = [open_or_create_image(p) for p in image_paths]
    grid_im = images_list_to_grid_image(ims, if_draw_grid=True, n_rows=1)
    grid_im.save(os.path.join(tmp_path, f'grid_many_experiments_{dataset_types}_{hashlib.sha256(str(experiment_names).encode("utf-8")).hexdigest()}.png'))
    if kwargs.get('print_median_run_flops_and_accs', False): # this is for named models in the appendix
        assert len(experiment_names) == 1
        algo_to_seed_to_result = get_test_metrics(experiment_names[0], dataset_types[0], max_iter=max_iters[0], algo_name=algo_names[0])
        print(experiment_names[0], dataset_types[0], max_iters[0], algo_names[0])
        seed_to_result = algo_to_seed_to_result[algo_names[0]]
        metrics = seed_to_result[idx_median]
        flops = [int(x) for x in metrics['flops']]
        accs = metrics[dataset_type].tolist()
        print(f'{list(zip(flops, accs))=}')
    print_test_metrics_best_mean_and_std_many(experiment_names, dataset_types, max_iters, algo_names, target_runs=target_runs)


def combine_runs_make_image(experiment_path, algo_name_to_seed_to_image_path, dataset_type, out_img_name_lambda, **kwargs):
    image_paths = []
    print(f'{algo_name_to_seed_to_image_path=}')
    for seed_to_image_path in algo_name_to_seed_to_image_path.values():
        print(f'{seed_to_image_path=}')
        image_paths += list(seed_to_image_path.values())
    w, h = Image.open(image_paths[0]).size
    open_or_create_image = lambda path: Image.new(mode='RGB', size=(w, h)) if path is None else Image.open(path)
    ims = [open_or_create_image(p) for p in image_paths]
    grid_im = images_list_to_grid_image(ims, if_draw_grid=True, n_rows=len(algo_name_to_seed_to_image_path))
    grid_im.save(os.path.join(experiment_path, out_img_name_lambda(dataset_type)))


def compare_val_and_test(experiment_name, dataset_type='test', **kwargs):
    dataset_postfix = kwargs.get('dataset_postfix', '')
    utils.execute_func_for_all_runs_and_combine(experiment_name, eval_cumulative_pareto_front_single_run,
                                                func_combine=combine_runs_make_image,
                                                dataset_type=dataset_type,
                                                out_img_name_lambda=lambda dataset_type: f'grid_best_pareto_val_and_{dataset_type}{dataset_postfix}.png',
                                                **kwargs)


def read_cumulative_pareto_front_metrics_single_run(experiment_path, dataset_type='test', **kwargs):
    max_iter = kwargs.get('max_iter', 15) # 15 # 30
    max_iter_path = os.path.join(nsga_logs_path, experiment_path, f'iter_{max_iter}')

    if_post_hoc_ensemble = 'posthoc' in experiment_path and 'posthocheavy' not in experiment_path
    save_path_base = experiment_path if if_post_hoc_ensemble else max_iter_path
    save_output_path = os.path.join(save_path_base, f'best_pareto_val_and_{dataset_type}.json')

    if not Path(save_output_path).is_file():
        if if_post_hoc_ensemble:
            # the path will be different if ensemble was evaluated from stored outputs
            save_output_path = os.path.join(save_path_base, f'posthoc_ensemble_from_stored_filtered.yml')
            if not os.path.exists(save_output_path):
                save_output_path = os.path.join(save_path_base, f'posthoc_ensemble_from_stored.yml')
            loaded_data = yaml.safe_load(open(save_output_path))
        else:
            raise FileNotFoundError(save_output_path)
    else:
        loaded_data = json.load(open(save_output_path))
    loaded_data_acc = np.array(loaded_data[dataset_type])
    if dataset_type == 'val': # crutches beget crutches beget crutches...
        loaded_data_acc = 100 - loaded_data_acc
    return {'test': loaded_data_acc, 'flops': loaded_data['flops']}


def get_test_metrics(experiment_name, dataset_type='test', **kwargs):
    algo_name_to_seed_to_result = utils.execute_func_for_all_runs_and_combine(experiment_name,
                                                                              read_cumulative_pareto_front_metrics_single_run,
                                                                              dataset_type=dataset_type,
                                                                              **kwargs)
    return algo_name_to_seed_to_result


def get_test_metrics_best_mean_and_std(experiment_name, dataset_type='test', max_iter=15, algo_name=None, **kwargs):
    if 'logs_classification' not in experiment_name and 'segmentation_logs' not in experiment_name:
        algo_name_to_seed_to_result = utils.execute_func_for_all_runs_and_combine(experiment_name, read_cumulative_pareto_front_metrics_single_run,
                                                                   dataset_type=dataset_type, max_iter=max_iter, **kwargs, target_algos=algo_name)
        if algo_name is None:
            algo_name = list(algo_name_to_seed_to_result.keys())[0]
        seed_to_result = algo_name_to_seed_to_result[algo_name]
        test_metrics = seed_to_result.values()
    else:
        nsga_path = utils.NAT_PATH
        test_metrics = []
        for seed_dir in sorted(os.scandir(os.path.join(nsga_path, experiment_name)), key=lambda e: e.name):
            data = yaml.safe_load(open(os.path.join(nsga_path, experiment_name, seed_dir.name, 'data.yml')))
            test_metrics.append({'test': data['test'], 'flops': data['flops']})

    def mean_and_std_for_max(ar):
        best = [np.max(x) for x in ar]
        mean, std = np.mean(best), np.std(best)
        return mean, std
    def mean_and_std_for_last(ar):
        last = [x[-1] for x in ar]
        mean, std = np.mean(last), np.std(last)
        return mean, std
    def compute_hypervolume(dict_metrics):
        test = np.array(dict_metrics['test'])
        flops = np.array(dict_metrics['flops'])

        worst_top1_err, worst_flops = 40, 4000
        ref_pt = np.array([worst_top1_err, worst_flops])
        test_hv = utils.compute_hypervolume(ref_pt, np.column_stack([100 - test, flops]), if_increase_ref_pt=False)
        return test_hv


    def mean_and_std(ar):
        return np.mean(ar), np.std(ar)

    return {'test': mean_and_std_for_last([x['test'] for x in test_metrics]),
            'flops': mean_and_std_for_max([x['flops'] for x in test_metrics]),
            'hv': mean_and_std([compute_hypervolume(x) for x in test_metrics])
            }, len(test_metrics)

def print_test_metrics_best_mean_and_std(experiment_name, dataset_type='test', max_iter=15, algo_name=None, **kwargs):
    means_and_stds, n_seeds = get_test_metrics_best_mean_and_std(experiment_name, dataset_type, max_iter, algo_name, **kwargs)
    print(f'{experiment_name} ({dataset_type}): '
          f'{means_and_stds["hv"][0]:.3f} ± {means_and_stds["hv"][1]:.3f} ; '
          f'{means_and_stds["test"][0]:.2f} ± {means_and_stds["test"][1]:.2f} ; '
          f'{int(means_and_stds["flops"][0])} ± {int(means_and_stds["flops"][1])} ({n_seeds} seeds)'
          )

def print_test_metrics_best_mean_and_std_many(experiment_names, dataset_type, max_iters, algo_names, **kwargs):
    if not type(dataset_type) is list:
        dataset_type = [dataset_type] * len(experiment_names)
    if algo_names is None:
        algo_names = [None] * len(experiment_names)
    for experiment_name, dataset_type_cur, max_iter, algo_name in zip(experiment_names, dataset_type, max_iters, algo_names):
        print_test_metrics_best_mean_and_std(experiment_name, dataset_type_cur, max_iter, algo_name, **kwargs)
        # break

def compute_hypervolumes(experiment_name, dataset_type='test', max_iter=15, algo_name=None, **kwargs):
    if 'logs_classification' not in experiment_name and 'segmentation_logs' not in experiment_name:
        algo_name_to_seed_to_result = utils.execute_func_for_all_runs_and_combine(experiment_name,
                                                                                  read_cumulative_pareto_front_metrics_single_run,
                                                                                  dataset_type=dataset_type,
                                                                                  max_iter=max_iter, **kwargs,
                                                                                  target_algos=algo_name)
        if algo_name is None:
            algo_name = list(algo_name_to_seed_to_result.keys())[0]
        seed_to_result = algo_name_to_seed_to_result[algo_name]
        test_metrics = seed_to_result.values()
    else:
        nsga_path = utils.NAT_PATH
        test_metrics = []
        for seed_dir in sorted(os.scandir(os.path.join(nsga_path, experiment_name)), key=lambda e: e.name):
            data = yaml.safe_load(open(os.path.join(nsga_path, experiment_name, seed_dir.name, 'data.yml')))
            test_metrics.append({'test': data['test'], 'flops': data['flops']})

    def compute_hypervolume(dict_metrics):
        test = np.array(dict_metrics['test'])
        flops = np.array(dict_metrics['flops'])

        worst_top1_err, worst_flops = 40, 4000
        ref_pt = np.array([worst_top1_err, worst_flops])
        test_hv = utils.compute_hypervolume(ref_pt, np.column_stack([100 - test, flops]), if_increase_ref_pt=False)
        return test_hv
    if not kwargs.get('if_return_max_accs', False):
        return [compute_hypervolume(x) for x in test_metrics]
    else:
        return [compute_hypervolume(x) for x in test_metrics], [x['test'][-1] for x in test_metrics]

def print_hypervolumes_many(experiment_names, dataset_type, max_iters, algo_names, **kwargs):
    if not type(dataset_type) is list:
        dataset_type = [dataset_type] * len(experiment_names)
    if algo_names is None:
        algo_names = [None] * len(experiment_names)
    if not type(max_iters) is list:
        max_iters = [max_iters] * len(experiment_names)
    for experiment_name, dataset_type_cur, max_iter, algo_name in zip(experiment_names, dataset_type, max_iters, algo_names):
        hvs = compute_hypervolumes(experiment_name, dataset_type_cur, max_iter, algo_name, **kwargs)
        print(f'{experiment_name} HV: {hvs}')

def plot_hypervolumes_impact_n_supernets(experiment_names, dataset_type, max_iters, algo_names, supernet_numbers,
                                         set_xticks, label, **kwargs):
    if not type(dataset_type) is list:
        dataset_type = [dataset_type] * len(experiment_names)
    if algo_names is None:
        algo_names = [None] * len(experiment_names)
    if not type(max_iters) is list:
        max_iters = [max_iters] * len(experiment_names)

    means = []
    stds = []
    for experiment_name, dataset_type_cur, max_iter, algo_name in zip(experiment_names, dataset_type, max_iters, algo_names):
        hvs = compute_hypervolumes(experiment_name, dataset_type_cur, max_iter, algo_name, **kwargs)
        mean, std = np.mean(hvs), np.std(hvs)
        print(f'n seeds = {len(hvs)}')
        means.append(mean)
        stds.append(std)

    if set_xticks:
        plt.xticks(supernet_numbers)

    plt.errorbar(supernet_numbers, means, yerr=stds, capsize=5, label=label)

def get_hypervolumes_and_max_accs_for_stat_testing(experiment_names, dataset_type, max_iters, algo_names, **kwargs):
    if not type(dataset_type) is list:
        dataset_type = [dataset_type] * len(experiment_names)
    if algo_names is None:
        algo_names = [None] * len(experiment_names)
    if not type(max_iters) is list:
        max_iters = [max_iters] * len(experiment_names)

    all_hvs = []
    all_max_accs = []
    for experiment_name, dataset_type_cur, max_iter, algo_name in zip(experiment_names, dataset_type, max_iters, algo_names):
        hvs, max_accs = compute_hypervolumes(experiment_name, dataset_type_cur, max_iter, algo_name, if_return_max_accs=True, **kwargs)
        print(f'n seeds = {len(hvs)}')
        all_hvs.append(hvs)
        all_max_accs.append(max_accs)
    return all_hvs, all_max_accs