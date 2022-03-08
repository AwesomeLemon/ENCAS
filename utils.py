import atexit
import gzip
import logging
import math
import os
import random
import sys
import yaml
from ofa.utils import count_parameters, measure_net_latency
from pathlib import Path
from ptflops import get_model_complexity_info
from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from typing import List

import numpy as np
from collections import defaultdict

from PIL import Image, ImageDraw

import torch
import torch.nn.functional

from matplotlib import pyplot as plt

import io
import selectors
import subprocess

from networks.ofa_mbv3_my import OFAMobileNetV3My

# NAT_PATH = '/export/scratch3/aleksand/nsganetv2'
NAT_PATH = '/projects/0/einf2071/nsganetv2'
NAT_LOGS_PATH = os.path.join(NAT_PATH, 'logs')
NAT_DATA_PATH = os.path.join(NAT_PATH, 'data')

_alphabets = ['full_nat', 'full_nat_w12', 'full_nat_w10', 'full_alphanet', 'full_nat_proxyless',
              'full_alphanet_cascade2', 'full_nat_w12_cascade2',
              'full_nat_w12_cascade5', 'full_nat_w10_cascade5', 'full_alphanet_cascade5', 'full_nat_proxyless_cascade5']

alphabet_dict = {a: os.path.join(NAT_PATH, 'alphabets', f'{a}.txt') for a in _alphabets}

ss_name_to_supernet_path = {'ofa12': 'supernet_w1.2', 'ofa10': 'supernet_w1.0',
                            'alphanet': 'alphanet_pretrained.pth.tar',
                            'alphanet1': 'alphanet_pretrained.pth.tar',
                            'alphanet2': 'alphanet_pretrained.pth.tar',
                            'alphanet3': 'alphanet_pretrained.pth.tar',
                            'alphanet4': 'alphanet_pretrained.pth.tar',
                            'attn': 'attentive_nas_pretrained.pth.tar',
                            'proxyless': 'ofa_proxyless_d234_e346_k357_w1.3'}

threshold_gene_to_value = {i: 0.1*(i + 1) for i in range(10)}
threshold_gene_to_value_moregranular = {i: 0.02 * i for i in range(51)}


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau


def look_up_latency(net, lut, resolution=224):
    def _half(x, times=1):
        for _ in range(times):
            x = np.ceil(x / 2)
        return int(x)

    predicted_latency = 0

    # first_conv
    predicted_latency += lut.predict(
        'first_conv', [resolution, resolution, 3],
        [resolution // 2, resolution // 2, net.first_conv.out_channels])

    # final_expand_layer (only for MobileNet V3 models)
    input_resolution = _half(resolution, times=5)
    predicted_latency += lut.predict(
        'final_expand_layer',
        [input_resolution, input_resolution, net.final_expand_layer.in_channels],
        [input_resolution, input_resolution, net.final_expand_layer.out_channels]
    )

    # feature_mix_layer
    predicted_latency += lut.predict(
        'feature_mix_layer',
        [1, 1, net.feature_mix_layer.in_channels],
        [1, 1, net.feature_mix_layer.out_channels]
    )

    # classifier
    predicted_latency += lut.predict(
        'classifier',
        [net.classifier.in_features],
        [net.classifier.out_features]
    )

    # blocks
    fsize = _half(resolution)
    for block in net.blocks:
        idskip = 0 if block.config['shortcut'] is None else 1
        se = 1 if block.config['mobile_inverted_conv']['use_se'] else 0
        stride = block.config['mobile_inverted_conv']['stride']
        out_fz = _half(fsize) if stride > 1 else fsize
        block_latency = lut.predict(
            'MBConv',
            [fsize, fsize, block.config['mobile_inverted_conv']['in_channels']],
            [out_fz, out_fz, block.config['mobile_inverted_conv']['out_channels']],
            expand=block.config['mobile_inverted_conv']['expand_ratio'],
            kernel=block.config['mobile_inverted_conv']['kernel_size'],
            stride=stride, idskip=idskip, se=se
        )
        predicted_latency += block_latency
        fsize = out_fz

    return predicted_latency


def get_metric_complement(metric, if_segmentation=False):
    max_value = 100
    if if_segmentation:
        max_value = 1
    return max_value - metric

def fix_folder_names_imagenetv2():
    import os, glob

    for path in glob.glob('/export/scratch3/aleksand/data/imagenet/imagenetv2_all'):
        if os.path.isdir(path):
            for subpath in glob.glob(f'{path}/*'):
                dirname = subpath.split('/')[-1]
                os.rename(subpath, '/'.join(subpath.split('/')[:-1]) + '/' + dirname.zfill(4))

def compute_hypervolume(ref_pt, F, normalized=True, if_increase_ref_pt=True, if_input_already_pareto=False):
    # calculate hypervolume on the non-dominated set of F
    if not if_input_already_pareto:
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
    else:
        nd_F = F
    if if_increase_ref_pt:
        ref_pt = 1.01 * ref_pt
    hv = get_performance_indicator('hv', ref_point=ref_pt).calc(nd_F)
    if normalized:
        hv = hv / np.prod(ref_pt)
    return hv


class LoggerWriter:
    def __init__(self, log_fun):
        self.log_fun = log_fun
        self.buf = []
        self.is_tqdm_msg_fun = lambda msg: '%|' in msg

    def write(self, msg):
        is_tqdm = self.is_tqdm_msg_fun(msg)
        has_newline = msg.endswith('\n')
        if has_newline or is_tqdm:
            self.buf.append(msg)#.rstrip('\n'))
            self.log_fun(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass

    def close(self):
        self.log_fun.close()


def setup_logging(log_path):
    from importlib import reload
    reload(logging)
    logging.StreamHandler.terminator = ''  # don't add new line, I'll do it myself; this line affects both handlers
    stream_handler = logging.StreamHandler(sys.__stdout__)
    file_handler = logging.FileHandler(log_path, mode='a')
    # don't want a bazillion tqdm lines in the log:
    # file_handler.filter = lambda record: '%|' not in record.msg or '100%|' in record.msg
    file_handler.filter = lambda record: '[A' not in record.msg and ('%|' not in record.msg or '100%|' in record.msg)
    handlers = [
        file_handler,
        stream_handler]
    logging.basicConfig(level=logging.INFO,
                        # format='%(asctime)s %(message)s',
                        format='%(message)s',
                        handlers=handlers,
                        datefmt='%H:%M')
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

# https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
from types import SimpleNamespace
class RecursiveNamespace(SimpleNamespace):

  @staticmethod
  def map_entry(entry):
    if isinstance(entry, dict):
      return RecursiveNamespace(**entry)

    return entry

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    for key, val in kwargs.items():
      if type(val) == dict:
        setattr(self, key, RecursiveNamespace(**val))
      elif type(val) == list:
        setattr(self, key, list(map(self.map_entry, val)))

alphanet_config_str = '''
        use_v3_head: True
        resolutions: [192, 224, 256, 288]
        first_conv: 
            c: [16, 24]
            act_func: 'swish'
            s: 2
        mb1:
            c: [16, 24]
            d: [1, 2]
            k: [3, 5]
            t: [1]
            s: 1
            act_func: 'swish'
            se: False
        mb2:
            c: [24, 32]
            d: [3, 4, 5]
            k: [3, 5]
            t: [4, 5, 6]
            s: 2
            act_func: 'swish'
            se: False
        mb3:
            c: [32, 40] 
            d: [3, 4, 5, 6]
            k: [3, 5]
            t: [4, 5, 6]
            s: 2
            act_func: 'swish'
            se: True
        mb4:
            c: [64, 72] 
            d: [3, 4, 5, 6]
            k: [3, 5]
            t: [4, 5, 6]
            s: 2
            act_func: 'swish'
            se: False
        mb5:
            c: [112, 120, 128] 
            d: [3, 4, 5, 6, 7, 8]
            k: [3, 5]
            t: [4, 5, 6]
            s: 1
            act_func: 'swish'
            se: True
        mb6:
            c: [192, 200, 208, 216] 
            d: [3, 4, 5, 6, 7, 8]
            k: [3, 5]
            t: [6]
            s: 2
            act_func: 'swish'
            se: True
        mb7:
            c: [216, 224] 
            d: [1, 2]
            k: [3, 5]
            t: [6]
            s: 1
            act_func: 'swish'
            se: True
        last_conv:
            c: [1792, 1984]
            act_func: 'swish'
        '''


def images_list_to_grid_image(ims, if_rgba=False, if_draw_middle_line=False, if_draw_grid=False,
                              n_rows=None, n_cols=None):
    n_ims = len(ims)
    width, height = ims[0].size
    rows_num = math.floor(math.sqrt(n_ims)) if n_rows is None else n_rows
    cols_num = int(math.ceil(n_ims / rows_num)) if n_cols is None else n_cols
    new_im = Image.new('RGB' if not if_rgba else 'RGBA', (cols_num * width, rows_num * height))
    for j in range(n_ims):
        row = j // cols_num
        column = j - row * cols_num
        new_im.paste(ims[j], (column * width, row * height))
    if if_draw_middle_line or if_draw_grid:
        draw = ImageDraw.Draw(new_im)
        if if_draw_middle_line:
            draw.line((0, height // 2 * rows_num - 1, width * cols_num, height // 2 * rows_num - 1),
                      fill=(200, 100, 100, 255), width=1)
        if if_draw_grid:
            if rows_num > 1:
                for i in range(1, rows_num):
                    draw.line((0, height * i - 1, width * cols_num, height * i - 1), fill=(0, 0, 0, 255), width=5)
            if cols_num > 1:
                for i in range(1, cols_num):
                    draw.line((width * i - 1, 0, width * i - 1, height * rows_num), fill=(0, 0, 0, 255), width=5)

    return new_im


class CsvLogger():
    def __init__(self, path, name):
        Path(path).mkdir(exist_ok=True)
        self.full_path = os.path.join(path, name)
        self.columns = ['Evaluation', 'Time', 'Solution', 'Fitness']
        self.data = []
        self.f = open(self.full_path, 'w', buffering=100)
        self.f.write(' '.join(self.columns) + '\n')
        atexit.register(self.close_f)

    def log(self, values: List):
        values_str = ' '.join(str(v) for v in values) + '\n'
        # print(values_str)
        self.f.write(values_str)

    def close_f(self):
        self.f.close()

def capture_subprocess_output(subprocess_args):
    # taken from https://gist.github.com/nawatts/e2cdca610463200c12eac2a14efc0bfb
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = subprocess.Popen(subprocess_args,
                               bufsize=1,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True,
                               # env=dict(os.environ, OMP_NUM_THREADS='9')
                               )

    # Create callback function for process output
    buf = io.StringIO()
    def handle_output(stream, mask):
        # Because the process' output is line buffered, there's only ever one
        # line to read when this function is called
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = (return_code == 0)

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return output

def set_seed(seed):
    print(f'Setting random seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def execute_func_for_all_runs_and_combine(experiment_name, func, func_combine=None, **kwargs):
    experiment_path = os.path.join(NAT_LOGS_PATH, experiment_name)
    algo_names = []
    algo_name_to_seed_to_result = defaultdict(dict)
    target_algos = kwargs.get('target_algos', None) # useful for debugging
    target_runs = kwargs.get('target_runs', None) # useful for debugging
    # print(f'{target_algos=}, {target_runs=}')
    for f in reversed(sorted(os.scandir(experiment_path), key=lambda e: e.name)):
        if not f.is_dir():
            continue
        name_cur = f.name
        if target_algos is not None and name_cur not in target_algos:
            continue
        algo_names.append(name_cur)
        for run_folder in os.scandir(f.path):
            if not run_folder.is_dir():
                continue
            run_idx = int(run_folder.name)
            if target_runs is not None and run_idx not in target_runs:
                continue
            run_path = os.path.join(experiment_path, name_cur, str(run_idx))
            out = func(run_path, run_idx=run_idx, **kwargs)
            algo_name_to_seed_to_result[name_cur][run_idx] = out

    if func_combine:
        return func_combine(experiment_path, algo_name_to_seed_to_result, experiment_name=experiment_name, **kwargs)

    return algo_name_to_seed_to_result

def save_gz(path, data):
    f = gzip.GzipFile(path, "w")
    np.save(file=f, arr=data)
    f.close()
    print(f'{path} saved')


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR

def show_im_from_torch_tensor(t):
    im = t.permute(1, 2, 0).numpy()
    plt.imshow(im * np.array([0.24703233, 0.24348505, 0.26158768]) + np.array([0.49139968, 0.48215827, 0.44653124]))
    plt.show()

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def transform_supernet_name_swa(supernet_name_in, swa):
    if supernet_name_in == 'alphanet_pretrained.pth.tar':
        return f'alphanet_pretrained_swa{swa}.pth.tar'
    elif supernet_name_in == 'attentive_nas_pretrained.pth.tar':
        return f'attentive_nas_pretrained_swa{swa}.pth.tar'
    elif 'supernet_w1' in supernet_name_in:
        return supernet_name_in + f'_swa{swa}'
    elif 'ofa_proxyless' in supernet_name_in:
        return supernet_name_in + f'_swa{swa}'
    else:
        return 'noop'


class LatencyEstimator(object):
    """
    Modified from https://github.com/mit-han-lab/proxylessnas/blob/
    f273683a77c4df082dd11cc963b07fc3613079a0/search/utils/latency_estimator.py#L29
    """
    def __init__(self, fname):
        # fname = download_url(url, overwrite=True)

        with open(fname, 'r') as fp:
            self.lut = yaml.safe_load(fp, yaml.SafeLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, expand=None,
                kernel=None, stride=None, idskip=None, se=None):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `first_conv`: The initial stem 3x3 conv with stride 2
                2. `final_expand_layer`: (Only for MobileNet-V3)
                    The upsample 1x1 conv that increases num_filters by 6 times + GAP.
                3. 'feature_mix_layer':
                    The upsample 1x1 conv that increase num_filters to num_features + torch.squeeze
                3. `classifier`: fully connected linear layer (num_features to num_classes)
                4. `MBConv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        :param se: indicate whether has squeeze-and-excitation
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input),
                 'output:%s' % self.repr_shape(output), ]
        if ltype in ('MBConv',):
            assert None not in (expand, kernel, stride, idskip, se)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel,
                      'stride:%d' % stride, 'idskip:%d' % idskip, 'se:%d' % se]
        key = '-'.join(infos)
        return self.lut[key]['mean']


def parse_string_list(string):
    if isinstance(string, str):
        # convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
        return list(map(int, string[1:-1].split()))
    else:
        return string


def pad_none(x, depth, max_depth):
    new_x, counter = [], 0
    for d in depth:
        for _ in range(d):
            new_x.append(x[counter])
            counter += 1
        if d < max_depth:
            new_x += [None] * (max_depth - d)
    return new_x


def validate_config(config, max_depth=4):
    kernel_size, exp_ratio, depth = config['ks'], config['e'], config['d']

    if isinstance(kernel_size, str): kernel_size = parse_string_list(kernel_size)
    if isinstance(exp_ratio, str): exp_ratio = parse_string_list(exp_ratio)
    if isinstance(depth, str): depth = parse_string_list(depth)

    assert (isinstance(kernel_size, list) or isinstance(kernel_size, int))
    assert (isinstance(exp_ratio, list) or isinstance(exp_ratio, int))
    assert isinstance(depth, list)

    if len(kernel_size) < len(depth) * max_depth:
        kernel_size = pad_none(kernel_size, depth, max_depth)
    if len(exp_ratio) < len(depth) * max_depth:
        exp_ratio = pad_none(exp_ratio, depth, max_depth)

    # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
    res = {'ks': kernel_size, 'e': exp_ratio, 'd': depth}
    if 'r' in config:
        res['r'] = config['r']
    if 'w' in config:
        res['w'] = config['w']
    else:
        res['w'] = 1.0
    if 'position' in config:
        res['position'] = config['position']
    if 'threshold' in config:
        res['threshold'] = config['threshold']
    return res


if __name__ == '__main__':
    fix_folder_names_imagenetv2()
    sys.exit()


def get_net_info(net, data_shape, measure_latency=None, print_info=True, clean=False, lut=None,
                 if_dont_sum=False):
    def inner(net_cur, data_shape):
        net_info = {}
        if isinstance(net_cur, torch.nn.DataParallel):
            net_cur = net_cur.module

        net_info['params'] = count_parameters(net_cur)
        net_info['flops'] = get_model_complexity_info(net_cur, (data_shape[0], data_shape[1], data_shape[2]),
                                                      print_per_layer_stat=False, as_strings=False, verbose=False)[0]

        latency_types = [] if measure_latency is None else measure_latency.split('#')
        for l_type in latency_types:
            if l_type == 'flops':
                continue  # already calculated above
            if lut is not None and l_type in lut:
                latency_estimator = LatencyEstimator(lut[l_type])
                latency = look_up_latency(net_cur, latency_estimator, data_shape[2])
                measured_latency = None
            else:
                latency, measured_latency = measure_net_latency(
                    net_cur, l_type, fast=False, input_shape=data_shape, clean=clean)
            net_info['%s latency' % l_type] = {'val': latency, 'hist': measured_latency}

        if print_info:
            print('Total training params: %.2fM' % (net_info['params'] / 1e6))
            print('Total FLOPs: %.2fM' % (net_info['flops'] / 1e6))
            for l_type in latency_types:
                print('Estimated %s latency: %.3fms' % (l_type, net_info['%s latency' % l_type]['val']))

        gpu_latency, cpu_latency = None, None
        for k in net_info.keys():
            if 'gpu' in k:
                gpu_latency = np.round(net_info[k]['val'], 2)
            if 'cpu' in k:
                cpu_latency = np.round(net_info[k]['val'], 2)

        return {'params': np.round(net_info['params'] / 1e6, 2),
                'flops': np.round(net_info['flops'] / 1e6, 2),
                'gpu': gpu_latency, 'cpu': cpu_latency}

    if not isinstance(net, list): # if not an ensemble, just calculate it
        return inner(net, data_shape)
    # if an ensemble, need to sum properly
    data_shapes = [(data_shape[0], s1, s2) for s1, s2 in zip(data_shape[1], data_shape[2])]
    results = [inner(net_cur, d_s) for net_cur, d_s in zip(net, data_shapes)]
    res_final = {} # sum everything, keep None as None
    for k, v in results[0].items():
        if not if_dont_sum:
            res_final[k] = v
            for res_i in results[1:]:
                if v is None:
                    continue
                res_final[k] += res_i[k]
        else:
            res_final[k] = [v]
            for res_i in results[1:]:
                if v is None:
                    continue
                res_final[k] += [res_i[k]]
    return res_final


class SupernetworkWrapper:
    def __init__(self,
                 n_classes=1000,
                 model_path='./data/ofa_mbv3_d234_e346_k357_w1.0',
                 engine_class_to_use=OFAMobileNetV3My, **kwargs):
        from nat import NAT
        self.dataset_name = kwargs['dataset']
        self.search_space_name = kwargs['search_space_name']
        engine_lambda = NAT.make_lambda_for_engine_creation(engine_class_to_use, n_classes, False,
                                                            self.dataset_name, self.search_space_name)
        self.engine, _ = engine_lambda(model_path, None, to_cuda=False, if_create_optimizer=False)

    def sample(self, config):
        if self.search_space_name == 'ofa':
            config = validate_config(config)
        self.engine.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'], w=config['w'])
        subnet = self.engine.get_active_subnet(preserve_weight=True)
        return subnet, config
