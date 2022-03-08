import copy

import utils
from collections import defaultdict

import pandas as pd
import timm
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from os.path import join

import numpy as np

from ofa.utils import AverageMeter, accuracy
from timm.data import create_dataset, create_loader, resolve_data_config
from tqdm import tqdm

import torch
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

path_timm_csv = os.path.join(utils.NAT_DATA_PATH, 'timm-results-imagenet.csv')
df_timm = pd.read_csv(path_timm_csv)

def compute_outputs_single_network(net_name, data_provider_kwargs, dataset_type, **kwargs):
    model = timm.create_model(net_name, pretrained=True)
    model.cuda()
    model.eval()
    df_row = df_timm[df_timm['model'] == net_name]
    image_size = int(df_row['img_size'].values[0])

    # here I use timm loader to get exactly the results reported in the repo
    if 'val' in dataset_type:
        split_name = 'imagenetv2_all'
    elif 'test' in dataset_type:
        split_name = 'val' # "validation" of ImageNet is used for test
    dataset = create_dataset(
        root=data_provider_kwargs['data'], name='', split=split_name,
        download=False, load_bytes=False, class_map='')
    args_for_data_config = {'model': 'beit_base_patch16_224', 'img_size': None,
                             'input_size': None, 'crop_pct': None, 'mean': None, 'std': None, 'interpolation': '',
                            'num_classes': 1000, 'class_map': '', 'gp': None, 'pretrained': True,
                            'test_pool': False, 'no_prefetcher': False, 'pin_mem': False, 'channels_last': False,
                            'tf_preprocessing': False, 'use_ema': False, 'torchscript': False, 'legacy_jit': False,
                            'prefetcher': True}
    data_config = resolve_data_config(args_for_data_config, model=model, use_test_size=True, verbose=True)

    crop_pct = data_config['crop_pct']
    loader = create_loader(dataset,
                           input_size=data_config['input_size'],
                           batch_size=data_provider_kwargs['vld_batch_size'],
                           use_prefetcher=True,
                           interpolation=data_config['interpolation'],
                           mean=data_config['mean'],
                           std=data_config['std'],
                           num_workers=data_provider_kwargs['n_workers'],
                           crop_pct=crop_pct,
                           pin_memory=True,
                           tf_preprocessing=False)
    n_batches = len(loader)
    # model = torch.nn.DataParallel(model)

    metric_dict = defaultdict(lambda: AverageMeter())
    outputs_to_return = []
    with tqdm(total=n_batches, desc=dataset_type, ncols=130) as t, torch.no_grad():
        for i, (images, labels, *_) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()

            with torch.cuda.amp.autocast():
                output = model(images)
            outputs_to_return.append(output.detach().cpu())
            acc1 = accuracy(output, labels, topk=(1,))[0].item()
            metric_dict['acc'].update(acc1, output.size(0))

            t.set_postfix({**{key: metric_dict[key].avg for key in metric_dict},
                           'img_size': images.size(2)})
            t.update(1)

    outputs_to_return = torch.cat(outputs_to_return, dim=0)

    if True:
        flops = FlopCountAnalysis(model.cuda(), torch.randn(1, 3, image_size, image_size).cuda())
        metric_dict['flops'] = flops.total() / 10 ** 6
    else:
        # used this to double-check the results - they are consistent between the libraries
        flops = get_model_complexity_info(model.cuda(), (3, image_size, image_size),
                                                  print_per_layer_stat=False, as_strings=False, verbose=False)[0]
        metric_dict['flops'] = flops / 10 ** 6
    return outputs_to_return, metric_dict


def store_outputs_many_networks(net_names, data_provider_kwargs, dataset_type, dir_name, **kwargs):
    save_path_base = join(utils.NAT_LOGS_PATH, dir_name)
    Path(save_path_base).mkdir(exist_ok=True)
    postfix = kwargs.get('postfix', '')
    out_folder_path = join(save_path_base, 'pretrained')
    Path(out_folder_path).mkdir(exist_ok=True) # need to create all the dirs in the hierarchy
    out_folder_path = join(out_folder_path, '0')
    Path(out_folder_path).mkdir(exist_ok=True)
    out_folder_path = join(out_folder_path, f'output_distrs_{dataset_type}{postfix}')
    Path(out_folder_path).mkdir(exist_ok=True)

    info_dict_path = os.path.join(out_folder_path, f'info.json')
    out_folder_logits_path = join(save_path_base, 'pretrained', '0', f'logit_gaps_{dataset_type}{postfix}')
    Path(out_folder_logits_path).mkdir(exist_ok=True)

    process_pool = ProcessPoolExecutor(max_workers=1)
    futures = []
    accs_all = []
    flops_all = []

    for i, net_name in enumerate(net_names):
        print(f'{net_name=}')
        if 'efficientnet' not in net_name: # large effnets use a lot of VRAM => use smaller batch
            logits, metric_dict = compute_outputs_single_network(net_name, data_provider_kwargs, dataset_type)
        else:
            data_provider_kwargs_smaller_batch = copy.deepcopy(data_provider_kwargs)
            data_provider_kwargs_smaller_batch['vld_batch_size'] = 20
            logits, metric_dict = compute_outputs_single_network(net_name, data_provider_kwargs_smaller_batch, dataset_type)

        logits_float32 = torch.tensor(logits, dtype=torch.float32)
        acc, flops = metric_dict['acc'].avg, metric_dict['flops']
        accs_all.append(acc)
        flops_all.append(flops)
        two_max_values = logits_float32.topk(k=2, dim=-1).values
        logit_gap = two_max_values[:, 0] - two_max_values[:, 1]
        future = process_pool.submit(utils.save_gz, path=os.path.join(out_folder_logits_path, f'{i}.npy.gz'),
                                     data=logit_gap.numpy().astype(np.float16))
        futures.append(future)

        outputs = torch.softmax(logits_float32, dim=-1)
        future = process_pool.submit(utils.save_gz, path=os.path.join(out_folder_path, f'{i}.npy.gz'),
                                     data=outputs.numpy().astype(np.float16))
        futures.append(future)

    info_dict = {dataset_type: accs_all, 'flops': flops_all, 'net_names': net_names}
    json.dump(info_dict, open(info_dict_path, 'w'))

    for f in futures:
        f.result()  # wait on everything


if __name__ == '__main__':
    IMAGENET_PATH = '/projects/0/einf2071/data/imagenet/' #'/export/scratch2/aleksand/data/imagenet/'
    # You can set the download location of the model checkpoints like this:
    # torch.hub.set_dir('/export/scratch2/aleksand/torch_hub/')
    # torch.hub.set_dir('/projects/0/einf2071/torch_hub/')
    all_timm_models_without_regnetz = ['beit_large_patch16_512', 'beit_large_patch16_384', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'beit_large_patch16_224', 'swin_large_patch4_window12_384', 'vit_large_patch16_384', 'tf_efficientnet_b7_ns', 'beit_base_patch16_384', 'cait_m48_448', 'tf_efficientnet_b6_ns', 'swin_base_patch4_window12_384', 'tf_efficientnetv2_xl_in21ft1k', 'swin_large_patch4_window7_224', 'tf_efficientnetv2_l_in21ft1k', 'vit_large_r50_s32_384', 'dm_nfnet_f6', 'tf_efficientnet_b5_ns', 'cait_m36_384', 'vit_base_patch16_384', 'xcit_large_24_p8_384_dist', 'vit_large_patch16_224', 'xcit_medium_24_p8_384_dist', 'dm_nfnet_f5', 'xcit_large_24_p16_384_dist', 'dm_nfnet_f4', 'tf_efficientnetv2_m_in21ft1k', 'xcit_small_24_p8_384_dist', 'dm_nfnet_f3', 'tf_efficientnetv2_l', 'cait_s36_384', 'ig_resnext101_32x48d', 'xcit_medium_24_p16_384_dist', 'deit_base_distilled_patch16_384', 'xcit_large_24_p8_224_dist', 'tf_efficientnet_b8_ap', 'tf_efficientnet_b8', 'swin_base_patch4_window7_224', 'beit_base_patch16_224', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b7_ap', 'xcit_small_24_p16_384_dist', 'ig_resnext101_32x32d', 'xcit_small_12_p8_384_dist', 'xcit_medium_24_p8_224_dist', 'dm_nfnet_f2', 'tf_efficientnetv2_m', 'cait_s24_384', 'resnetrs420', 'ecaresnet269d', 'vit_base_r50_s16_384', 'resnetv2_152x4_bitm', 'tf_efficientnet_b7', 'xcit_large_24_p16_224_dist', 'xcit_small_24_p8_224_dist', 'efficientnetv2_rw_m', 'tf_efficientnet_b6_ap', 'eca_nfnet_l2', 'xcit_small_12_p16_384_dist', 'resnetrs350', 'dm_nfnet_f1', 'vit_base_patch16_224', 'resnest269e', 'resnetv2_152x2_bitm', 'vit_large_r50_s32_224', 'resnetrs270', 'resnetv2_101x3_bitm', 'resmlp_big_24_224_in22ft1k', 'xcit_large_24_p8_224', 'seresnet152d', 'tf_efficientnetv2_s_in21ft1k', 'xcit_medium_24_p16_224_dist', 'vit_base_patch16_224_miil', 'swsl_resnext101_32x8d', 'tf_efficientnet_b5_ap', 'xcit_small_12_p8_224_dist', 'crossvit_18_dagger_408', 'ig_resnext101_32x16d', 'pit_b_distilled_224', 'tf_efficientnet_b6', 'resnetrs200', 'cait_xs24_384', 'vit_small_r26_s32_384', 'tf_efficientnet_b3_ns', 'eca_nfnet_l1', 'resnetv2_50x3_bitm', 'resnet200d', 'tf_efficientnetv2_s', 'xcit_small_24_p16_224_dist', 'resnest200e', 'xcit_small_24_p8_224', 'resnetv2_152x2_bit_teacher_384', 'efficientnetv2_rw_s', 'crossvit_15_dagger_408', 'tf_efficientnet_b5', 'vit_small_patch16_384', 'xcit_tiny_24_p8_384_dist', 'xcit_medium_24_p8_224', 'resnetrs152', 'regnety_160', 'twins_svt_large', 'resnet152d', 'resmlp_big_24_distilled_224', 'jx_nest_base', 'cait_s24_224', 'efficientnet_b4', 'deit_base_distilled_patch16_224', 'dm_nfnet_f0', 'swsl_resnext101_32x16d', 'xcit_small_12_p16_224_dist', 'vit_base_patch32_384', 'xcit_small_12_p8_224', 'tf_efficientnet_b4_ap', 'swsl_resnext101_32x4d', 'swin_small_patch4_window7_224', 'twins_pcpvt_large', 'twins_svt_base', 'jx_nest_small', 'deit_base_patch16_384', 'tresnet_m', 'tresnet_xl_448', 'tf_efficientnet_b4', 'resnet101d', 'resnetv2_152x2_bit_teacher', 'xcit_large_24_p16_224', 'resnest101e', 'resnetv2_50x1_bit_distilled', 'pnasnet5large', 'nfnet_l0', 'regnety_032', 'twins_pcpvt_base', 'ig_resnext101_32x8d', 'nasnetalarge', 'xcit_medium_24_p16_224', 'eca_nfnet_l0', 'levit_384', 'xcit_small_24_p16_224', 'xcit_tiny_24_p8_224_dist', 'xcit_tiny_24_p16_384_dist', 'resnet61q', 'crossvit_18_dagger_240', 'gc_efficientnetv2_rw_t', 'pit_b_224', 'crossvit_18_240', 'xcit_tiny_12_p8_384_dist', 'tf_efficientnet_b2_ns', 'resnet51q', 'ecaresnet50t', 'efficientnetv2_rw_t', 'resnetv2_101x1_bitm', 'crossvit_15_dagger_240', 'coat_lite_small', 'mixer_b16_224_miil', 'resnetrs101', 'convit_base', 'tresnet_l_448', 'efficientnet_b3', 'crossvit_base_240', 'cait_xxs36_384', 'ecaresnet101d', 'swsl_resnext50_32x4d', 'visformer_small', 'tresnet_xl', 'resnetv2_101', 'pit_s_distilled_224', 'deit_base_patch16_224', 'xcit_small_12_p16_224', 'tf_efficientnetv2_b3', 'xcit_tiny_24_p8_224', 'ssl_resnext101_32x16d', 'vit_small_r26_s32_224', 'tf_efficientnet_b3_ap', 'tresnet_m_448', 'twins_svt_small', 'tf_efficientnet_b3', 'rexnet_200', 'ssl_resnext101_32x8d', 'halonet50ts', 'tf_efficientnet_lite4', 'crossvit_15_240', 'halo2botnet50ts_256', 'tnt_s_patch16_224', 'vit_large_patch32_384', 'levit_256', 'tresnet_l', 'wide_resnet50_2', 'jx_nest_tiny', 'lamhalobotnet50ts_256', 'convit_small', 'swin_tiny_patch4_window7_224', 'vit_small_patch16_224', 'tf_efficientnet_b1_ns', 'convmixer_1536_20', 'gernet_l', 'legacy_senet154', 'efficientnet_el', 'coat_mini', 'seresnext50_32x4d', 'gluon_senet154', 'xcit_tiny_12_p8_224_dist', 'deit_small_distilled_patch16_224', 'lambda_resnet50ts', 'resmlp_36_distilled_224', 'swsl_resnet50', 'resnest50d_4s2x40d', 'twins_pcpvt_small', 'pit_s_224', 'haloregnetz_b', 'resmlp_big_24_224', 'crossvit_small_240', 'gluon_resnet152_v1s', 'resnest50d_1s4x24d', 'sehalonet33ts', 'resnest50d', 'cait_xxs24_384', 'xcit_tiny_12_p16_384_dist', 'gcresnet50t', 'ssl_resnext101_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'efficientnet_b3_pruned', 'ecaresnet101d_pruned', 'regnety_320', 'resmlp_24_distilled_224', 'vit_base_patch32_224', 'gernet_m', 'nf_resnet50', 'gluon_resnext101_64x4d', 'ecaresnet50d', 'efficientnet_b2', 'gcresnext50ts', 'resnet50d', 'repvgg_b3', 'vit_small_patch32_384', 'gluon_resnet152_v1d', 'mixnet_xl', 'xcit_tiny_24_p16_224_dist', 'ecaresnetlight', 'inception_resnet_v2', 'resnetv2_50', 'gluon_resnet101_v1d', 'regnety_120', 'resnet50', 'seresnet33ts', 'resnetv2_50x1_bitm', 'gluon_resnext101_32x4d', 'rexnet_150', 'tf_efficientnet_b2_ap', 'ssl_resnext50_32x4d', 'efficientnet_el_pruned', 'gluon_resnet101_v1s', 'regnetx_320', 'tf_efficientnet_el', 'seresnet50', 'vit_base_patch16_sam_224', 'legacy_seresnext101_32x4d', 'repvgg_b3g4', 'tf_efficientnetv2_b2', 'dpn107', 'convmixer_768_32', 'inception_v4', 'skresnext50_32x4d', 'eca_resnet33ts', 'gcresnet33ts', 'tf_efficientnet_b2', 'cspresnext50', 'cspdarknet53', 'dpn92', 'ens_adv_inception_resnet_v2', 'gluon_seresnext50_32x4d', 'gluon_resnet152_v1c', 'efficientnet_b2_pruned', 'xception71', 'regnety_080', 'resnetrs50', 'deit_small_patch16_224', 'levit_192', 'ecaresnet26t', 'regnetx_160', 'dpn131', 'tf_efficientnet_lite3', 'resnext50_32x4d', 'resmlp_36_224', 'cait_xxs36_224', 'regnety_064', 'xcit_tiny_12_p8_224', 'ecaresnet50d_pruned', 'gluon_xception65', 'gluon_resnet152_v1b', 'resnext50d_32x4d', 'dpn98', 'gmlp_s16_224', 'regnetx_120', 'cspresnet50', 'xception65', 'gluon_resnet101_v1c', 'rexnet_130', 'tf_efficientnetv2_b1', 'hrnet_w64', 'xcit_tiny_24_p16_224', 'dla102x2', 'resmlp_24_224', 'repvgg_b2g4', 'gluon_resnext50_32x4d', 'tf_efficientnet_cc_b1_8e', 'hrnet_w48', 'resnext101_32x8d', 'ese_vovnet39b', 'gluon_resnet101_v1b', 'resnetblur50', 'nf_regnet_b1', 'pit_xs_distilled_224', 'tf_efficientnet_b1_ap', 'eca_botnext26ts_256', 'botnet26t_256', 'efficientnet_em', 'ssl_resnet50', 'regnety_040', 'regnetx_080', 'dpn68b', 'resnet33ts', 'res2net101_26w_4s', 'halonet26t', 'lambda_resnet26t', 'coat_lite_mini', 'legacy_seresnext50_32x4d', 'gluon_resnet50_v1d', 'regnetx_064', 'xception', 'resnet32ts', 'res2net50_26w_8s', 'mixnet_l', 'lambda_resnet26rpt_256', 'hrnet_w40', 'hrnet_w44', 'wide_resnet101_2', 'eca_halonext26ts', 'tf_efficientnet_b1', 'efficientnet_b1', 'gluon_inception_v3', 'repvgg_b2', 'tf_mixnet_l', 'dla169', 'gluon_resnet50_v1s', 'legacy_seresnet152', 'tf_efficientnet_b0_ns', 'xcit_tiny_12_p16_224_dist', 'res2net50_26w_6s', 'xception41', 'dla102x', 'regnetx_040', 'resnest26d', 'levit_128', 'dla60_res2net', 'vit_tiny_patch16_384', 'hrnet_w32', 'dla60_res2next', 'coat_tiny', 'selecsls60b', 'legacy_seresnet101', 'repvgg_b1', 'cait_xxs24_224', 'tf_efficientnetv2_b0', 'tv_resnet152', 'bat_resnext26ts', 'efficientnet_b1_pruned', 'dla60x', 'res2next50', 'hrnet_w30', 'pit_xs_224', 'regnetx_032', 'tf_efficientnet_em', 'res2net50_14w_8s', 'hardcorenas_f', 'efficientnet_es', 'gmixer_24_224', 'dla102', 'gluon_resnet50_v1c', 'res2net50_26w_4s', 'selecsls60', 'seresnext26t_32x4d', 'resmlp_12_distilled_224', 'mobilenetv3_large_100_miil', 'tf_efficientnet_cc_b0_8e', 'resnet26t', 'regnety_016', 'tf_inception_v3', 'rexnet_100', 'seresnext26ts', 'gcresnext26ts', 'xcit_nano_12_p8_384_dist', 'hardcorenas_e', 'efficientnet_b0', 'legacy_seresnet50', 'tv_resnext50_32x4d', 'repvgg_b1g4', 'seresnext26d_32x4d', 'adv_inception_v3', 'gluon_resnet50_v1b', 'res2net50_48w_2s', 'coat_lite_tiny', 'tf_efficientnet_lite2', 'inception_v3', 'eca_resnext26ts', 'hardcorenas_d', 'tv_resnet101', 'densenet161', 'tf_efficientnet_cc_b0_4e', 'densenet201', 'mobilenetv2_120d', 'mixnet_m', 'selecsls42b', 'xcit_tiny_12_p16_224', 'resnet34d', 'tf_efficientnet_b0_ap', 'legacy_seresnext26_32x4d', 'hardcorenas_c', 'dla60', 'crossvit_9_dagger_240', 'tf_mixnet_m', 'regnetx_016', 'convmixer_1024_20_ks9_p14', 'skresnet34', 'gernet_s', 'tf_efficientnet_b0', 'ese_vovnet19b_dw', 'resnext26ts', 'hrnet_w18', 'resnet26d', 'tf_efficientnet_lite1', 'resmlp_12_224', 'mixer_b16_224', 'tf_efficientnet_es', 'densenetblur121d', 'levit_128s', 'hardcorenas_b', 'mobilenetv2_140', 'repvgg_a2', 'xcit_nano_12_p8_224_dist', 'regnety_008', 'dpn68', 'tv_resnet50', 'vit_small_patch32_224', 'mixnet_s', 'vit_tiny_r_s16_p8_384', 'hardcorenas_a', 'densenet169', 'mobilenetv3_large_100', 'tf_mixnet_s', 'mobilenetv3_rw', 'densenet121', 'tf_mobilenetv3_large_100', 'resnest14d', 'efficientnet_lite0', 'xcit_nano_12_p16_384_dist', 'vit_tiny_patch16_224', 'semnasnet_100', 'resnet26', 'regnety_006', 'repvgg_b0', 'fbnetc_100', 'resnet34', 'hrnet_w18_small_v2', 'regnetx_008', 'mobilenetv2_110d', 'efficientnet_es_pruned', 'tf_efficientnet_lite0', 'legacy_seresnet34', 'tv_densenet121', 'mnasnet_100', 'dla34', 'gluon_resnet34_v1b', 'pit_ti_distilled_224', 'deit_tiny_distilled_patch16_224', 'vgg19_bn', 'spnasnet_100', 'regnety_004', 'ghostnet_100', 'crossvit_9_240', 'xcit_nano_12_p8_224', 'regnetx_006', 'vit_base_patch32_sam_224', 'tf_mobilenetv3_large_075', 'vgg16_bn', 'crossvit_tiny_240', 'tv_resnet34', 'swsl_resnet18', 'convit_tiny', 'skresnet18', 'mobilenetv2_100', 'pit_ti_224', 'ssl_resnet18', 'regnetx_004', 'vgg19', 'hrnet_w18_small', 'xcit_nano_12_p16_224_dist', 'resnet18d', 'tf_mobilenetv3_large_minimal_100', 'deit_tiny_patch16_224', 'mixer_l16_224', 'vit_tiny_r_s16_p8_224', 'legacy_seresnet18', 'vgg16', 'vgg13_bn', 'gluon_resnet18_v1b', 'vgg11_bn', 'regnety_002', 'xcit_nano_12_p16_224', 'vgg13', 'resnet18', 'vgg11', 'regnetx_002', 'tf_mobilenetv3_small_100', 'dla60x_c', 'dla46x_c', 'tf_mobilenetv3_small_075', 'dla46_c', 'tf_mobilenetv3_small_minimal_100']

    # it's convenient to download all the models in advance:
    for net_name in reversed(all_timm_models_without_regnetz):
        try:
            model = timm.create_model(net_name, pretrained=True, num_classes=1000)
            del model
        except:
            print(f'Failed {net_name}')

    data_provider_kwargs = {'data': IMAGENET_PATH, 'dataset': 'imagenet', 'n_workers': 8, 'vld_batch_size': 128}
    # Snellius:
    store_outputs_many_networks(all_timm_models_without_regnetz, data_provider_kwargs, 'val', 'timm_all')
    store_outputs_many_networks(all_timm_models_without_regnetz, data_provider_kwargs, 'test', 'timm_all')