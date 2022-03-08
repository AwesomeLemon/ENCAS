import os
import torch

import utils


def swa(run_path, iters, supernet_name_in, supernet_name_out):
    checkpoint_paths = [os.path.join(run_path, f'iter_{i}', supernet_name_in) for i in iters]
    # read checkpoints
    checkpoints = [torch.load(p, map_location='cpu') for p in checkpoint_paths]
    state_dicts = [c['model_state_dict'] for c in checkpoints]
    # for all keys, average
    out_state_dict = {}
    for k, v in state_dicts[0].items():
        if v.data.dtype in [torch.int, torch.long]:
            out_state_dict[k] = state_dicts[-1][k] #num batches tracked => makes sense to take the last value
            continue

        for state_dict in state_dicts:
            if k in out_state_dict:
                out_state_dict[k] += state_dict[k]
            else:
                out_state_dict[k] = state_dict[k]

        out_state_dict[k] /= len(state_dicts)
    # save the result
    out_checkpoint = checkpoints[-1]
    out_checkpoint['model_state_dict'] = out_state_dict

    torch.save(out_checkpoint, os.path.join(run_path, f'iter_{iters[-1]}', supernet_name_out))

def swa_for_whole_experiment(experiment_name, iters, supernet_name_in, target_runs=None):
    nsga_logs_path = utils.NAT_LOGS_PATH
    experiment_path = os.path.join(nsga_logs_path, experiment_name)
    algo_names = []
    image_paths = []
    for f in reversed(sorted(os.scandir(experiment_path), key=lambda e: e.name)):
        if not f.is_dir():
            continue
        name_cur = f.name
        algo_names.append(name_cur)
        for run_folder in os.scandir(f.path):
            if not run_folder.is_dir():
                continue
            run_idx = int(run_folder.name)
            if target_runs is not None and run_idx not in target_runs:
                continue
            run_path = os.path.join(experiment_path, name_cur, str(run_idx))
            im_path = swa(run_path, iters, supernet_name_in, utils.transform_supernet_name_swa(supernet_name_in, len(iters)))
            image_paths.append(im_path)