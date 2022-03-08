import numpy as np
import torch
import yaml

from run_manager import get_run_config
from utils import NAT_PATH, NAT_DATA_PATH
import os

'''
Store labels of val & test sets for all the datasets.
To not specify all data-related information again, simply use any NAT config that uses the appropriate dataset
'''

def get_val_and_test_labels(nat_config_name):
    nat_config = yaml.safe_load(open(os.path.join(NAT_PATH, 'configs_nat', nat_config_name), 'r'))
    run_config = get_run_config(dataset=nat_config['dataset'], data_path=nat_config['data'],
                                train_batch_size=nat_config['trn_batch_size'], total_epochs=0,
                                test_batch_size=nat_config['vld_batch_size'], n_worker=4,
                                cutout_size=nat_config['cutout_size'], image_size=32,
                                valid_size=nat_config['vld_size'], dataset_name=nat_config['dataset'])
    run_config.valid_loader.collate_fn.set_resolutions([32]) # images are not used, but need to set some size.
    lbls_val = [b[1] for b in run_config.valid_loader]
    lbls_val = torch.cat(lbls_val).detach().numpy()

    lbls_test = [b[1] for b in run_config.test_loader]
    lbls_test = torch.cat(lbls_test).detach().numpy()

    return lbls_val, lbls_test

if __name__ == '__main__':
    lbls_val, lbls_test = get_val_and_test_labels('cifar10_r0_ofa10_sep.yml')
    np.save(os.path.join(NAT_DATA_PATH, 'labels_cifar10_val10000'), lbls_val)
    np.save(os.path.join(NAT_DATA_PATH, 'labels_cifar10_test'), lbls_test)

    lbls_val, lbls_test = get_val_and_test_labels('cifar100_r0_ofa10_sep.yml')
    np.save(os.path.join(NAT_DATA_PATH, 'labels_cifar100_val10000'), lbls_val)
    np.save(os.path.join(NAT_DATA_PATH, 'labels_cifar100_test'), lbls_test)

    lbls_val, lbls_test = get_val_and_test_labels('imagenet_r0_ofa10_sep.yml')
    np.save(os.path.join(NAT_DATA_PATH, 'labels_imagenet_val20683'), lbls_val)
    np.save(os.path.join(NAT_DATA_PATH, 'labels_imagenet_test'), lbls_test)