import itertools
import os
import glob
from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import utils
from nat import NAT
import yaml

def compute_hypervolumes_over_time(run_path, **kwargs):
    csv_path = glob.glob(os.path.join(run_path, '*.csv'))[0]
    save_path = os.path.join(run_path, kwargs['out_name_hvs'])
    if os.path.exists(save_path) and not kwargs.get('overwrite', False):
        print(f'Loaded the already-computed values for {run_path}')
        return yaml.safe_load(open(save_path, 'r'))#, Loader=yaml.BaseLoader)

    df = pd.read_csv(os.path.join(csv_path), sep=' ')
    fitnesses = np.array([[float(t) for t in x.strip('()').split(',')] for x in df.iloc[:, -1]], dtype=np.float)

    if 'gomea.csv' in csv_path:
        fitnesses *= -1

    worst_top1_err, worst_flops = 40, 4000
    ref_pt = np.array([worst_top1_err, worst_flops])

    pareto = np.array([])

    hvs = []
    if_pareto_changed = False
    hvs_step = 100

    for i, (top1_err, flops) in enumerate(fitnesses):
        print(i)
        if len(pareto) == 0:
            pareto = np.array([[top1_err, flops]])
            continue
        idx_dominate0 = pareto[:, 0] < top1_err
        idx_dominate1 = pareto[:, 1] < flops
        is_dominated = np.any(idx_dominate0 * idx_dominate1)
        if not is_dominated:
            idx_dominated0 = pareto[:, 0] >= top1_err
            idx_dominated1 = pareto[:, 1] >= flops
            idx_not_dominated = (1 - idx_dominated0 * idx_dominated1).astype(np.bool)
            pareto = pareto[idx_not_dominated]
            pareto = np.append(pareto, [[top1_err, flops]], axis=0)
            if_pareto_changed = True
            print(f'{pareto.shape=}')
        if (i + 1) % hvs_step == 0:
            if if_pareto_changed:
                hv = utils.compute_hypervolume(ref_pt, pareto, if_increase_ref_pt=False, if_input_already_pareto=True)
                if_pareto_changed = False
            hvs.append(float(hv))

        # if (i + 1) % 60000 == 0:
        #     plt.plot(pareto[:, 1], utils.get_metric_complement(pareto[:, 0], False), 'o')
        #     plt.title(str(hv))
        #     plt.show()
    out = hvs_step, hvs
    yaml.safe_dump(out, open(save_path, 'w'), default_flow_style=None)
    return out

def plot_hvs_with_stds(experiment_path, algo_name_to_seed_to_result, **kwargs):
    clist = matplotlib.rcParams['axes.prop_cycle']
    cgen = itertools.cycle(clist)

    map_algo_name = {'random': 'Random search', 'mo-gomea': 'MO-GOMEA'}

    for algo_name, seed_to_result in algo_name_to_seed_to_result.items():
        n_seeds = len(seed_to_result)
        all_hvs_cur = np.zeros((n_seeds, len(seed_to_result[0][1])))
        for i in range(n_seeds):
            hvs_step, hvs_cur = seed_to_result[i]
            all_hvs_cur[i] = hvs_cur

        mean_hvs_cur = np.mean(all_hvs_cur, axis=0)
        std_hvs_cur = np.std(all_hvs_cur, axis=0)

        x_ticks = np.arange(1, all_hvs_cur.shape[-1] + 1) * hvs_step
        cur_color = next(cgen)['color']
        plt.plot(x_ticks, mean_hvs_cur, label=map_algo_name[algo_name], c=cur_color)
        plt.fill_between(x_ticks, mean_hvs_cur - std_hvs_cur, mean_hvs_cur + std_hvs_cur,
                         facecolor=cur_color + '50')
        plt.legend()
        plt.xlabel('Evaluations - log scale')
        plt.xscale('log')

    if kwargs.get('show_title', True):
        plt.title(f'Hypervolume, {kwargs["experiment_name"]}')

    plt_path = kwargs.get('plt_path', os.path.join(experiment_path, 'hv_over_time.png'))
    plt.savefig(plt_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def plot_hvs_experiment(experiment_name, **kwargs):
    utils.execute_func_for_all_runs_and_combine(experiment_name, compute_hypervolumes_over_time, plot_hvs_with_stds, **kwargs)

if __name__ == '__main__':
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    # plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    tmp_path = os.path.join(utils.NAT_PATH, '.tmp')

    # Fig. 12
    plot_hvs_experiment('posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
                        out_name_hvs='hvs.yml', target_algos=['random', 'mo-gomea'], show_title=False,
                        plt_path=os.path.join(tmp_path, 'vs_random_hv_cifar10.pdf'), overwrite=False)
    plot_hvs_experiment('posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
                        out_name_hvs='hvs.yml', target_algos=['random', 'mo-gomea'], show_title=False,
                        plt_path=os.path.join(tmp_path, 'vs_random_hv_cifar100.pdf'), overwrite=False)
    plot_hvs_experiment('posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
                        out_name_hvs='hvs.yml', target_algos=['random', 'mo-gomea'], show_title=False,
                        plt_path=os.path.join(tmp_path, 'vs_random_hv_imagenet.pdf'), overwrite=False)