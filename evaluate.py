import time
from collections import defaultdict
import json

import torch
import numpy as np
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

from networks.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from networks.ofa_mbv3_my import OFAMobileNetV3My
from networks.proxyless_my import OFAProxylessNASNetsMy
from run_manager import get_run_config
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1

from run_manager.run_manager_my import RunManagerMy
from utils import set_seed, get_net_info, SupernetworkWrapper


def evaluate_many_configs(supernet_folder_path, configs, if_test=False, config_msunas=None, **kwargs):
    accs = []
    args = {k: v[0] for k, v in default_kwargs.items()}
    if config_msunas is not None:
        for key in ['data', 'dataset', 'n_classes', 'trn_batch_size', 'vld_batch_size',
                    'vld_size', 'n_workers', 'sec_obj']:
            args[key] = config_msunas[key]
    args['pass_subnet_config_directly'] = True
    args['test'] = if_test
    args['cutout_size'] = config_msunas.get('cutout_size', 32)
    args['reset_running_statistics'] = True
    args.update(kwargs)
    if 'thresholds' not in args:
        args['thresholds'] = None

    info_keys_to_return = []
    if 'info_keys_to_return' in kwargs:
        info_keys_to_return = kwargs['info_keys_to_return']
    info_keys_to_return_2_values = defaultdict(list)
    args['info_keys_to_return'] = info_keys_to_return

    args['supernet_path'] = supernet_folder_path
    args['search_space_name'] = kwargs['search_space_name']
    args['ensemble_ss_names'] = kwargs['ensemble_ss_names']

    # a hack for speed: reuse run_config for all the subnetworks evaluated
    #                   it should change nothing because the subnet architecture is the only
    #                   thing changing in the _for_ loop
    run_config = kwargs.get('run_config', None)

    for config in configs:
        args['subnet'] = config
        args['run_config'] = run_config

        info = _evaluate_one_config(args)
        top1_error = info['top1']
        run_config = info['run_config']
        accs.append(top1_error)
        for key in info_keys_to_return:
            info_keys_to_return_2_values[key].append(info[key])
    if len(info_keys_to_return) > 0:
        return accs, info_keys_to_return_2_values
    return accs


def _evaluate_one_config(args):
    set_seed(args['random_seed'])

    preproc_alphanet = False
    if args['pass_subnet_config_directly']:
        config = args['subnet']
    else:
        config = json.load(open(args['subnet']))
    if args['search_space_name'] == 'reproduce_nat':
        if config['w'] == 1.0:
            evaluator = SupernetworkWrapper(n_classes=args['n_classes'], model_path=args['supernet_path'][0],
                                            engine_class_to_use=OFAMobileNetV3My, dataset=args['dataset'],
                                            search_space_name='ofa')
        else:
            evaluator = SupernetworkWrapper(n_classes=args['n_classes'], model_path=args['supernet_path'][1],
                                            engine_class_to_use=OFAMobileNetV3My, dataset=args['dataset'],
                                            search_space_name='ofa')

        subnet, _ = evaluator.sample(config)
        subnet = subnet.cuda()
        resolution = config['r']
    elif args['search_space_name'] == 'ensemble':
        ensemble_ss_names = args['ensemble_ss_names']
        supernet_paths = args['supernet_path']

        ss_name_to_class = {'alphanet': AttentiveNasDynamicModel, 'ofa': OFAMobileNetV3My,
                            'proxyless': OFAProxylessNASNetsMy}

        # some ensembles have missing members which are represented by config that is None
        # for ENCAS, also need to remove thresholds
        if args['thresholds'] is None:
            filtered = filter(lambda conf_p_e: conf_p_e[0] is not None, zip(config, supernet_paths, ensemble_ss_names))
            config, supernet_paths, ensemble_ss_names = list(zip(*filtered))
        else:
            filtered = filter(lambda conf_p_e_t: conf_p_e_t[0] is not None, zip(config, supernet_paths, ensemble_ss_names, args['thresholds']))
            config, supernet_paths, ensemble_ss_names, thresholds = list(zip(*filtered))
            args['thresholds'] = thresholds
        print(f'{supernet_paths=}')

        classes_to_use = [ss_name_to_class[ss_name] for ss_name in ensemble_ss_names]

        evaluators = [SupernetworkWrapper(n_classes=args['n_classes'], model_path=supernet_path,
                                          engine_class_to_use=encoder_class, dataset=args['dataset'],
                                          search_space_name=ss_name)
                      for supernet_path, ss_name, encoder_class in zip(supernet_paths, ensemble_ss_names, classes_to_use)]
        subnet = [e.sample(c)[0] for e, c in zip(evaluators, config)]
        resolution = [conf['r'] for conf in config]

        # If normal ENCAS, thresholds are already provided. Otherwise:
        if args['thresholds'] is None:
            if 'threshold' in config[0]: # ENCAS-joint
                # (but the condition is also satisfied if ENCAS was run on subnets extracted from ENCAS-joint)
                # (that was a bug, now ENCAS won't execute this code no matter which subnets it uses)
                thresholds = [c['threshold'] for c in config]
                positions = [c['position'] for c in config]

                idx = np.argsort(positions)[::-1]
                # don' need to sort positions_list itself?
                thresholds = np.array(thresholds)[idx].tolist()
                resolution = np.array(resolution)[idx].tolist()
                subnet = np.array(subnet)[idx].tolist()

                args['thresholds'] = thresholds
            else: # not a cascade => can rearrange order
                idx = np.argsort(resolution)[::-1]
                resolution = np.array(resolution)[idx].tolist()
                subnet = np.array(subnet)[idx].tolist()

        preproc_alphanet ='alphanet' in ensemble_ss_names

    return _evaluate_one_model(
        subnet, data_path=args['data'], dataset=args['dataset'], resolution=resolution,
        trn_batch_size=args['trn_batch_size'], vld_batch_size=args['vld_batch_size'], num_workers=args['n_workers'],
        valid_size=args['vld_size'], is_test=args['test'], measure_latency=args['latency'],
        no_logs=(not args['verbose']), reset_running_statistics=args['reset_running_statistics'],
        run_config=args.get('run_config', None), sec_obj=args['sec_obj'],
        info_keys_to_return=args['info_keys_to_return'], cutout_size=args['cutout_size'], thresholds=args['thresholds'],
        if_use_logit_gaps=args['if_use_logit_gaps'], preproc_alphanet=preproc_alphanet)


def _evaluate_one_model(subnet, data_path, dataset='imagenet', resolution=224, trn_batch_size=128,
                        vld_batch_size=250, num_workers=4, valid_size=None, is_test=True,
                        measure_latency=None, no_logs=False, reset_running_statistics=True,
                        run_config=None, sec_obj='flops', info_keys_to_return=(), cutout_size=None, thresholds=None, if_use_logit_gaps=False,
                        preproc_alphanet=False):
    info = get_net_info(subnet, (3, resolution, resolution), measure_latency=measure_latency,
                        print_info=False, clean=True, if_dont_sum=thresholds is not None)
    print(f"{info['flops']=}")

    if_return_logit_gaps = 'logit_gaps' in info_keys_to_return
    validation_kwargs = {'if_return_outputs': 'output_distr' in info_keys_to_return,
                         'if_return_logit_gaps': if_return_logit_gaps}
    resolution_is_list = type(resolution) is list
    if resolution_is_list:
        validation_kwargs['resolutions_list'] = resolution_list = resolution
        resolution = max(resolution)  # collators need max resolution; will downsample in the val loop
        # Actually, collators need the first resolution, which in a cascade won't be the largest one
        validation_kwargs['thresholds'] = thresholds
        if thresholds is not None:
            resolution = resolution_list[0]
    validation_kwargs['if_use_logit_gaps'] = if_use_logit_gaps

    if run_config is None:
        run_config = get_run_config(dataset=dataset, data_path=data_path, image_size=resolution, n_epochs=0,
            train_batch_size=trn_batch_size, test_batch_size=vld_batch_size, n_worker=num_workers,
            valid_size=valid_size, total_epochs=0, dataset_name=dataset, cutout_size=cutout_size,
            preproc_alphanet=preproc_alphanet)

    data_provider = run_config.data_provider
    data_provider.collator_train.set_resolutions([resolution])
    data_provider.collator_subtrain.set_resolutions([resolution])
    run_config.valid_loader.collate_fn.set_resolutions([resolution])
    run_config.test_loader.collate_fn.set_resolutions([resolution])

    data_provider.assign_active_img_size(resolution)

    run_manager = RunManagerMy(subnet, run_config, no_gpu=False, sec_obj=sec_obj)
    if reset_running_statistics:
        # same subset size & batch size as during evaluation in training
        if not run_manager.is_ensemble:
            data_provider.collator_subtrain.set_resolutions([resolution])
            data_loader = run_config.random_sub_train_loader(2304 * 6, vld_batch_size, resolution)
            set_running_statistics(subnet, data_loader)
        else:
            for i_net, net_cur in enumerate(subnet):
                print(f'Resetting BNs for network {i_net}')
                st = time.time()
                data_provider.collator_subtrain.set_resolutions([resolution_list[i_net]])
                mul_due_to_logit_gaps = 6 # logit gaps differ a lot when comparing 3 and 1
                data_loader = run_config.random_sub_train_loader(2304 * mul_due_to_logit_gaps, vld_batch_size,
                                                                 resolution_list[i_net])
                net_cur.cuda()
                if hasattr(net_cur, 'reset_running_stats_for_calibration'):  # alphanet & attentiveNAS
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        net_cur.set_bn_param(0.1, 1e-5)
                        net_cur.eval()
                        net_cur.reset_running_stats_for_calibration()
                        for images, _ in data_loader:
                            images = images.cuda(non_blocking=True)
                            out = net_cur(images)
                        images.cpu(), out.cpu()
                        del images, out
                else:
                    set_running_statistics(net_cur, data_loader)
                ed = time.time()
                print(f'BN resetting time for {i_net}: {ed - st}')
        print('BNs reset')


    loss, dict_of_metrics = run_manager.validate(net=subnet, is_test=is_test, no_logs=no_logs, **validation_kwargs)

    top1 = dict_of_metrics['top1']
    info['loss'], info['top1'] = loss, top1

    if thresholds is not None:
        n_not_predicted_per_stage = dict_of_metrics['n_not_predicted_per_stage']
        flops_per_stage = info['flops']
        if is_test:
            data_loader = run_config.test_loader
        else:
            data_loader = run_config.valid_loader
        n_images_total = len(data_loader.dataset)
        print(f'{n_images_total=}, {n_not_predicted_per_stage=}')
        true_flops = flops_per_stage[0] + sum([n_not_predicted / n_images_total * flops for (n_not_predicted, flops) in
                                               zip(n_not_predicted_per_stage, flops_per_stage[1:])])
        info['flops'] = true_flops
        print(f'{thresholds=}')

    print(info)
    info['run_config'] = run_config  # a hack
    for k in info_keys_to_return:
        if k not in info:
            info[k] = dict_of_metrics[k]
    return info

# these are mostly irrelevant, will be overwritten. TODO: remove
default_kwargs = {
    'n_gpus': [1, 'total number of available gpus'],
    'gpu': [1, 'number of gpus per evaluation job'],
    'data': ['/export/scratch3/aleksand/data/CIFAR/', 'location of the data corpus'],
    'dataset': ['cifar10', 'name of the dataset [imagenet, cifar10, cifar100, ...]'],
    'n_classes': [10, 'number of classes of the given dataset'],
    'n_workers': [8, 'number of workers for dataloaders'],
    'vld_size': [5000, 'validation set size, randomly sampled from training set'],
    'trn_batch_size': [96, 'train batch size for training'],
    'vld_batch_size': [96, 'validation batch size'],
    'n_epochs': [0, 'n epochs to train'],
    'drop_rate': [0.2, 'dropout rate'],
    'drop_connect_rate': [0.0, ''],
    'resolution': [224, 'resolution'],
    'supernet_path': ['/export/scratch3/aleksand/nsganetv2/data/ofa_mbv3_d234_e346_k357_w1.0',
                      'path to supernet'],
    'subnet': ['', 'location of a json file of ks, e, d, and e'],
    'pass_subnet_config_directly': [False, 'Pass config as object instead of file path'],
    'config': ['', 'location of a json file of specific model declaration; not relevant for me'],
    'init': [None, 'location of initial weight to load'],
    'test': [False, 'if evaluate on test set'],
    'verbose': [True, ''],
    'save': [None, ''],
    'reset_running_statistics': [False, 'reset_running_statistics for BN'],
    'latency': [None, 'latency measurement settings (gpu64#cpu)'],
    'random_seed': [42, 'random seed'],
    'teacher_model': [None, ''],
}