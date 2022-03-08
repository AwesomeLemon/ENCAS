import numpy as np
import random

import utils


class OFASearchSpace:
    def __init__(self, alphabet='2', **kwargs):
        self.name = 'ofa'
        self.num_blocks = 5
        self.encoded_length = 22 #needed for decoding an ensemble
        self.if_cascade = False
        self.positions = [None]
        self.thresholds = [None]

        if alphabet == 'full_nat':
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
            self.resolution = list(range(192, 257, 4))  # input image resolutions
            self.width_mult = [1.0, 1.2]
        elif alphabet == 'full_nat_w10':
            self.kernel_size = [3, 5, 7]
            self.exp_ratio = [3, 4, 6]
            self.depth = [2, 3, 4]
            self.resolution = list(range(192, 257, 4))
            self.width_mult = [1.0]
        elif alphabet == 'full_nat_w12':
            self.kernel_size = [3, 5, 7]
            self.exp_ratio = [3, 4, 6]
            self.depth = [2, 3, 4]
            self.resolution = list(range(192, 257, 4))
            self.width_mult = [1.2]
        elif alphabet in ['full_nat_w12_cascade2', 'full_nat_w12_cascade5']: # size of the cascade is passed in kwargs
            self.if_cascade = True
            self.cascade_size = kwargs['ensemble_size']
            self.positions = list(range(self.cascade_size))
            self.n_thredsholds = len(utils.threshold_gene_to_value)
            self.threshold_value_to_gene = {v: k for k, v in utils.threshold_gene_to_value.items()}
            self.thresholds = [utils.threshold_gene_to_value[i] for i in range(self.n_thredsholds)]
            self.encoded_length += 2 # position, threshold

            self.kernel_size = [3, 5, 7]
            self.exp_ratio = [3, 4, 6]
            self.depth = [2, 3, 4]
            self.resolution = list(range(192, 257, 4))
            self.width_mult = [1.2]
        elif alphabet == 'full_nat_w10_cascade5':
            self.if_cascade = True
            self.cascade_size = kwargs['ensemble_size']
            self.positions = list(range(self.cascade_size))
            self.n_thredsholds = len(utils.threshold_gene_to_value)
            self.threshold_value_to_gene = {v: k for k, v in utils.threshold_gene_to_value.items()}
            self.thresholds = [utils.threshold_gene_to_value[i] for i in range(self.n_thredsholds)]
            self.encoded_length += 2 # position, threshold

            self.kernel_size = [3, 5, 7]
            self.exp_ratio = [3, 4, 6]
            self.depth = [2, 3, 4]
            self.resolution = list(range(192, 257, 4))
            self.width_mult = [1.0]
        else:
            raise ValueError(f'Unknown alphabet "{alphabet}"')



    def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, r=None, w=None, p=None, t=None):
        """ randomly sample a architecture"""
        nb = self.num_blocks if nb is None else nb
        ks = self.kernel_size if ks is None else ks
        e = self.exp_ratio if e is None else e
        d = self.depth if d is None else d
        r = self.resolution if r is None else r
        w = self.width_mult if w is None else w
        p = self.positions if p is None else p
        t = self.thresholds if t is None else t

        data = []
        for n in range(n_samples):
            # first sample layers
            depth = np.random.choice(d, nb, replace=True).tolist()
            # then sample kernel size, expansion rate and resolution
            kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
            exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()
            resolution = int(np.random.choice(r))
            width = np.random.choice(w)

            arch = {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution, 'w':width}
            if self.if_cascade:
                arch['position'] = random.choice(p)
                arch['threshold'] = random.choice(t)

            while arch in data:
                # first sample layers
                depth = np.random.choice(d, nb, replace=True).tolist()
                # then sample kernel size, expansion rate and resolution
                kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
                exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()
                resolution = int(np.random.choice(r))
                width = np.random.choice(w)
                arch = {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution, 'w': width}
                if self.if_cascade:
                    arch['position'] = random.choice(p)
                    arch['threshold'] = random.choice(t)

            data.append(arch)
        return data

    def initialize(self, n_doe):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        # print('Achtung! Add best NAT subnet to the initialization!')
        data = [
            self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)], d=[min(self.depth)],
                        r=[min(self.resolution)], w=[min(self.width_mult)], p=[min(self.positions)],
                        t=[min(self.thresholds)])[0],
            self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)], d=[max(self.depth)],
                        r=[max(self.resolution)], w=[max(self.width_mult)], p=[max(self.positions)],
                        t=[max(self.thresholds)])[0],
            # self.sample(1, ks= [7, 7, 7, 7, 7, 3, 7, 5, 7, 7, 7, 3, 7, 7, 7, 3],
            #             e= [3, 3, 6, 4, 6, 4, 3, 3, 6, 4, 6, 6, 6, 6, 3, 3],
            #             d=[2, 2, 4, 4, 4],
            #             r=[224], w=[1.2])[0]
        ]
        data.extend(self.sample(n_samples=n_doe - 2))
        return data


    def pad_zero(self, x, depth):
        # pad zeros to make bit-string of equal length
        new_x, counter = [], 0
        for d in depth:
            for _ in range(d):
                new_x.append(x[counter])
                counter += 1
            if d < max(self.depth):
                new_x += [0] * (max(self.depth) - d)
        return new_x

    def encode(self, config):
        """
        values of architecture parameters -> their indices
        """
        layer_choices = {'[3 3]': 1, '[3 5]': 2, '[3 7]': 3,
                         '[4 3]': 4, '[4 5]': 5, '[4 7]': 6,
                         '[6 3]': 7, '[6 5]': 8, '[6 7]': 9, '[None None]': 0}

        kernel_size = self.pad_zero(config['ks'], config['d'])
        exp_ratio = self.pad_zero(config['e'], config['d'])

        r = np.where(np.array(self.resolution) == config["r"])[0][0]
        w = np.where(np.array(self.width_mult) == config["w"])[0][0]

        layers = [0] * (self.num_blocks * max(self.depth))
        for i, d in enumerate(config['d']):
            for j in range(d):
                idx = i * max(self.depth) + j
                key = '[{} {}]'.format(exp_ratio[idx], kernel_size[idx])
                layers[idx] = layer_choices[key]

        layers = [r] + [w] + layers
        if self.if_cascade:
            pos = config['position']
            th = config['threshold']
            layers += [pos, self.threshold_value_to_gene[th]]
        return layers

    def decode(self, _layers):
        """
        indices of values of architecture parameters -> actual values
        """
        if type(_layers) is np.ndarray:
            _layers = _layers.flatten()

        cfg_choices = {1: (3, 3), 2: (3, 5), 3: (3, 7),
                       4: (4, 3), 5: (4, 5), 6: (4, 7),
                       7: (6, 3), 8: (6, 5), 9: (6, 7), 0: (None, None)}

        depth, kernel_size, exp_ratio = [], [], []

        resolution, width_mult = self.resolution[_layers[0]], self.width_mult[_layers[1]]

        d = 0
        layers = _layers[2:]
        if self.if_cascade:
            pos = int(layers[-2])
            th = float(utils.threshold_gene_to_value[layers[-1]])
            layers = layers[:-2]
        for i, l in enumerate(layers):
            e, ks = cfg_choices[l]
            if (ks is not None) and (e is not None):
                kernel_size.append(ks)
                exp_ratio.append(e)
                d += 1
            if (i + 1) % max(self.depth) == 0:
                if l != 0 and layers[i - 1] == 0:
                    # non-skip layer cannot follow skip layer
                    # we know the first 2 layers are non-skip, so we just need to check the 3rd one
                    # if it is 0, remove the current one
                    d -= 1
                    kernel_size = kernel_size[:-1]
                    exp_ratio = exp_ratio[:-1]
                depth.append(d)
                d = 0

        config = {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution, 'w': width_mult}
        if self.if_cascade:
            config['position'] = pos
            config['threshold'] = th
        return config