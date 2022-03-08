from copy import copy

import numpy as np
import yaml

import utils
from utils import RecursiveNamespace, alphanet_config_str


class AlphaNetSearchSpace:
    def __init__(self, alphabet, **kwargs):
        self.supernet_config = RecursiveNamespace(**yaml.safe_load(alphanet_config_str))
        self.supernet_config_dict = yaml.safe_load(alphanet_config_str)
        self.if_cascade = False

        if alphabet == 'full_alphanet':
            self.resolutions = [192, 224, 256, 288]
            self.min_config = [0] * 28
            self.max_config = [len(self.resolutions) - 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 2, 5, 1, 2, 3, 5, 1, 1, 1, 1, 1]
        elif alphabet in ['full_alphanet_cascade2', 'full_alphanet_cascade5']: # size of the cascade is passed in kwargs
            self.if_cascade = True
            cascade_size = kwargs['ensemble_size']
            self.positions = list(range(cascade_size))
            n_thredsholds = len(utils.threshold_gene_to_value)
            self.threshold_value_to_gene = {v: k for k, v in utils.threshold_gene_to_value.items()}

            self.resolutions = [192, 224, 256, 288]
            self.min_config = [0] * (28 + 2)
            self.max_config = [len(self.resolutions) - 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 2, 5, 1, 2, 3,
                               5, 1, 1, 1, 1, 1, cascade_size - 1, n_thredsholds - 1]
        else:
            raise NotImplementedError()

        self.name = 'alphanet'
        self.encoded_length = len(self.max_config) #needed for decoding an ensemble

    def sample(self, n_samples=1):
        data = []
        sampled_archive = set()
        for n in range(n_samples):
            # sample random encoding in range from all zeroes to max_config, then decode it
            sampled_encoding = [np.random.randint(lower_val, upper_val + 1)
                                for lower_val, upper_val in zip(self.min_config, self.max_config)]
            while tuple(sampled_encoding) in sampled_archive:
                sampled_encoding = [np.random.randint(lower_val, upper_val + 1)
                                    for lower_val, upper_val in zip(self.min_config, self.max_config)]

            data.append(self.decode(sampled_encoding))
            sampled_archive.add(tuple(sampled_encoding))
        return data

    def initialize(self, n_doe):
        # init with smallest & largest possible subnets + random ones
        data = [
            self.decode(self.min_config),
            self.decode(self.max_config)
        ]
        data.extend(self.sample(n_samples=n_doe - 2))
        return data

    def encode(self, config):
        """
        values to their indices
        """
        r = self.resolutions.index(config['r'])
        layers = [r]

        # first conv
        layers.append(self.supernet_config.first_conv.c.index(config['w'][0]))

        # blocks
        for i, (w, d, k, e) in enumerate(zip(config['w'][1:-1], config['d'], config['ks'], config['e'])):
            layers.append(self.supernet_config_dict[f'mb{i+1}']['c'].index(w))
            layers.append(self.supernet_config_dict[f'mb{i+1}']['d'].index(d))
            layers.append(self.supernet_config_dict[f'mb{i+1}']['k'].index(k))

            # blocks mb1, mb6, mb7 have a single possible value for expansion rates => don't encode
            if i not in [0, 5, 6]:
                layers.append(self.supernet_config_dict[f'mb{i + 1}']['t'].index(e))

        # last conv
        layers.append(self.supernet_config.last_conv.c.index(config['w'][-1]))

        if self.if_cascade:
            layers.append(int(config['position'])) # encoding == value itself
            layers.append(int(self.threshold_value_to_gene[config['threshold']]))

        return layers

    def decode(self, enc_conf):
        """
        transform list of choice indices to 4 lists of actual choices; all equal len except width that has 2 more values
        (note that variables with a single choice are not encoded; their values are simply added here)
        """
        if type(enc_conf) is np.ndarray:
            enc_conf = list(enc_conf.flatten())
        enc_conf = copy(enc_conf)

        depth, kernel_size, exp_ratio, width = [], [], [], []

        resolution = self.resolutions[enc_conf.pop(0)]

        # first conv
        width.append(self.supernet_config.first_conv.c[enc_conf.pop(0)])

        # blocks (code is not pretty, but I think writing it as a cycle would've been uglier)
        width.append(self.supernet_config.mb1.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb1.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb1.k[enc_conf.pop(0)])
        exp_ratio.append(1)

        width.append(self.supernet_config.mb2.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb2.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb2.k[enc_conf.pop(0)])
        exp_ratio.append(self.supernet_config.mb2.t[enc_conf.pop(0)])

        width.append(self.supernet_config.mb3.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb3.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb3.k[enc_conf.pop(0)])
        exp_ratio.append(self.supernet_config.mb3.t[enc_conf.pop(0)])

        width.append(self.supernet_config.mb4.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb4.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb4.k[enc_conf.pop(0)])
        exp_ratio.append(self.supernet_config.mb4.t[enc_conf.pop(0)])

        width.append(self.supernet_config.mb5.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb5.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb5.k[enc_conf.pop(0)])
        exp_ratio.append(self.supernet_config.mb5.t[enc_conf.pop(0)])

        width.append(self.supernet_config.mb6.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb6.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb6.k[enc_conf.pop(0)])
        exp_ratio.append(6)

        width.append(self.supernet_config.mb7.c[enc_conf.pop(0)])
        depth.append(self.supernet_config.mb7.d[enc_conf.pop(0)])
        kernel_size.append(self.supernet_config.mb7.k[enc_conf.pop(0)])
        exp_ratio.append(6)

        # last conv
        width.append(self.supernet_config.last_conv.c[enc_conf.pop(0)])

        config = {'r': resolution, 'w': width, 'd': depth, 'ks': kernel_size, 'e': exp_ratio}

        if self.if_cascade:
            config['position'] = int(enc_conf.pop(0))
            config['threshold'] = float(utils.threshold_gene_to_value[enc_conf.pop(0)])

        if len(enc_conf) != 0:
            raise AssertionError('not the whole config was used')

        return config

if __name__ == '__main__':
    ss = AlphaNetSearchSpace('full_alphanet')
    conf = {'r': 288, 'w': [24, 24, 32, 40, 72, 128, 216, 224, 1984], 'd': [2, 5, 6, 6, 8, 8, 2],
        'ks': [5, 5, 5, 5, 5, 5, 5], 'e': [1, 6, 6, 6, 6, 6, 6]}
    encoded = ss.encode(conf)
    print(f'{encoded=}')
    decoded = ss.decode(encoded)
    print(f'{decoded=}')

    conf = {'r': 288, 'w': [16, 16, 24, 32, 64, 112, 192, 216, 1792], 'd': [2, 5, 6, 6, 8, 8, 2],
        'ks': [5, 5, 5, 5, 5, 5, 5], 'e': [1, 6, 6, 6, 6, 6, 6]}
    encoded = ss.encode(conf)
    print(f'{encoded=}')
    decoded = ss.decode(encoded)
    print(f'{decoded=}')

    decoded_zeros = ss.decode([0] * 28)
    print(f'{decoded_zeros=}')