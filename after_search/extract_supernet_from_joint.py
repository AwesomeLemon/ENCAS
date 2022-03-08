import glob

import json
import numpy as np
import os
from os.path import join
from pathlib import Path
from shutil import copy

import re
import yaml

import utils
import utils_pareto
from utils import NAT_LOGS_PATH

def extract(experiment_name, out_experiment_name, idx_snet, if_joint_pareto_only=False, **kwargs): # idx_snet == idx of supernet to extract
    experiment_path = join(NAT_LOGS_PATH, experiment_name)
    out_path = join(NAT_LOGS_PATH, out_experiment_name)
    Path(out_path).mkdir(exist_ok=True)

    for f in reversed(sorted(os.scandir(experiment_path), key=lambda e: e.name)):
        if not f.is_dir():
            continue
        name_cur = f.name
        out_path_algo_cur = join(out_path, name_cur)
        Path(out_path_algo_cur).mkdir(exist_ok=True)

        for run_folder in os.scandir(f.path):
            if not run_folder.is_dir():
                continue
            run_idx = int(run_folder.name)
            run_path = join(experiment_path, name_cur, str(run_idx))
            run_path_new = join(out_path_algo_cur, str(run_idx))
            Path(run_path_new).mkdir(exist_ok=True)

            # copy & modify config_msunas
            msunas_config_new_path = join(run_path_new, 'config_msunas.yml')
            copy(join(run_path, 'config_msunas.yml'), msunas_config_new_path)
            msunas_config = yaml.safe_load(open(msunas_config_new_path, 'r'))
            for key in ['ensemble_ss_names', 'supernet_path', 'alphabet']:
                msunas_config[key] = [msunas_config[key][idx_snet]]
            yaml.dump(msunas_config, open(msunas_config_new_path, 'w'))

            # copy & modify iter_*.stats
            stats_paths = glob.glob(os.path.join(run_path, "iter_*.stats"))
            regex = re.compile(r'\d+')
            iters = np.array([int(regex.findall(p)[-1]) for p in stats_paths])
            idx = np.argsort(iters)
            iters = iters[idx]
            stats_paths = np.array(stats_paths)[idx].tolist()

            if if_joint_pareto_only:
                print(f'{iters[-1]=}')
                cfgs_jointpareto, _, _, iters_jointpareto = utils_pareto.get_best_pareto_up_and_including_iter(run_path, iters[-1])

            for it, p in enumerate(stats_paths):
                print(p)
                data = json.load(open(p, 'r'))
                data_new = {}
                for k, v in data.items():
                    if k != 'archive':
                        data_new[k] = v
                    else:
                        archive = v
                        new_archive = []
                        for (cfg_ensemble, top1_ens, flops_ens, top1s_and_flops_sep) in archive:
                            if if_joint_pareto_only:
                                cfgs_jointpareto_curiter = cfgs_jointpareto[iters_jointpareto == it]
                                if cfg_ensemble not in cfgs_jointpareto_curiter:
                                    continue

                            cfg = cfg_ensemble[idx_snet]
                            top1s_sep = top1s_and_flops_sep[0]
                            flops_sep = top1s_and_flops_sep[1]

                            new_archive_member = [[cfg], top1s_sep[idx_snet], flops_sep[idx_snet]]
                            new_archive.append(new_archive_member)
                        data_new['archive'] = new_archive
                # store data_new
                path_new = p.replace(run_path, run_path_new)
                with open(path_new, 'w') as handle:
                    json.dump(data_new, handle)

            # create iter_* folders, softlink weights
            # also softlink swa weights
            swa = kwargs.get('swa', None)
            for it in iters:
                it_path = join(run_path, f'iter_{it}')
                it_path_new = join(run_path_new, f'iter_{it}')
                Path(it_path_new).mkdir(exist_ok=True)
                supernet_name = os.path.basename(msunas_config['supernet_path'][0])
                if not os.path.exists(os.path.join(it_path_new, supernet_name)):
                    os.symlink(os.path.join(it_path, supernet_name), os.path.join(it_path_new, supernet_name))

            if swa is not None: # it will be stored in the last folder
                supernet_name = utils.transform_supernet_name_swa(os.path.basename(msunas_config['supernet_path'][0]), swa)
                if not os.path.exists(os.path.join(it_path_new, supernet_name)):
                    os.symlink(os.path.join(it_path, supernet_name), os.path.join(it_path_new, supernet_name))

def extract_all(experiment_name, out_name_suffixes, **kwargs):
    for i, out_name_suffix in enumerate(out_name_suffixes):
        extract(experiment_name, experiment_name + f'_EXTRACT_{out_name_suffix}', i, **kwargs)