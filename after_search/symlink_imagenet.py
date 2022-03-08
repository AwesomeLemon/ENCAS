import glob
import os

import utils


def create_symlinks(experiment_path, **kwargs):
    nsga_path = utils.NAT_LOGS_PATH
    full_path = os.path.join(nsga_path, experiment_path)
    files_to_symlink_all = ['supernet_w1.0', 'supernet_w1.2', 'ofa_proxyless_d234_e346_k357_w1.3',
                        'attentive_nas_pretrained.pth.tar', 'alphanet_pretrained.pth.tar']
    files_to_symlink_actual = []
    for f in files_to_symlink_all:
        if os.path.exists(os.path.join(full_path, f)):
            files_to_symlink_actual.append(f)
    for path in glob.glob(os.path.join(full_path, "iter_*/")):
        for f in files_to_symlink_actual:
            try:
                os.symlink(os.path.join(full_path, f), os.path.join(path, f))
            except FileExistsError:
                pass

if __name__ == '__main__':
    utils.execute_func_for_all_runs_and_combine('imagenet_v3_alpha_sep', create_symlinks)