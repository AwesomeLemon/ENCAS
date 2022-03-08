import random

import copy

import ctypes

import torch
import multiprocessing as mp
import numpy as np
from torchvision import transforms

from utils import onehot, rand_bbox, show_im_from_torch_tensor


class DynamicResolutionCollator:
    def __init__(self, n_resolutions_max, if_return_target_idx=True, if_cutmix=False, cutmix_kwargs=None):
        self.resolutions = mp.Array(ctypes.c_int, n_resolutions_max)
        self.n_resolutions_to_use = n_resolutions_max
        self.n_resolutions_max = n_resolutions_max
        self.resize_dict = {}
        self.if_return_target_idx = if_return_target_idx
        self.if_cutmix = if_cutmix
        self.prev_batch_for_cutmix = None
        self.cutmix_kwargs = cutmix_kwargs


    def set_info_for_transforms(self, resize_class_lambda, transforms_after_resize, transforms_pre_resize=[]):
        # this MUST be called before the dataloaders are actually used!
        # I would've put it in __init__, but I need to create collators before creating the dataprovider,
        #                   and these values are created only during creation of the dataprovider
        self.resize_class_lambda = resize_class_lambda
        self.transforms_after_resize = transforms_after_resize
        self.transforms_pre_resize = transforms_pre_resize

    def set_resolutions(self, resolutions):
        self.n_resolutions_to_use = len(resolutions)
        if self.n_resolutions_to_use > self.n_resolutions_max:
            raise ValueError('self.n_resolutions_to_use > self.n_resolutions_max')
        for i in range(self.n_resolutions_to_use):
            cur_res = resolutions[i]
            self.resolutions[i] = cur_res

    def __call__(self, batch):
        # don't need sync 'cause don't need to change the array of resolutions
        target_idx = np.random.choice(self.n_resolutions_to_use)
        target_res = self.resolutions[target_idx]
        if target_res not in self.resize_dict:
            self.resize_dict[target_res] = self.resize_class_lambda(target_res)
        cur_resize_op = self.resize_dict[target_res]

        transforms_composed = transforms.Compose(self.transforms_pre_resize + [cur_resize_op] + self.transforms_after_resize)
        imgs = [transforms_composed(img_n_label[0]) for img_n_label in batch]
        label = [img_n_label[1] for img_n_label in batch]

        if self.if_cutmix:
            cur_batch_before_cutmix = list(zip(copy.deepcopy(imgs), copy.deepcopy(label)))
            if self.prev_batch_for_cutmix is None: #this is the first batch
                self.prev_batch_for_cutmix = cur_batch_before_cutmix
            def cutmix(img, lbl):
                args = self.cutmix_kwargs
                lbl_onehot = onehot(args['n_classes'], lbl)
                if np.random.rand(1) > args['prob']:
                    return img, lbl_onehot

                rand_index = random.choice(range(len(self.prev_batch_for_cutmix)))
                img2, lbl2 = self.prev_batch_for_cutmix[rand_index]
                lbl2_onehot = onehot(args['n_classes'], lbl2)

                lam = np.random.beta(args['beta'], args['beta'])
                W, H = img.shape[-2:]
                W2, H2 = img2.shape[-2:]
                # my batches have different spatial sizes - that's the whole point of this collator!
                W, H = min(W, W2), min(H, H2)
                bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)
                img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                lbl_onehot = lbl_onehot * lam + lbl2_onehot * (1. - lam)

                return img, lbl_onehot

            img_n_label_cutmix = [cutmix(im, lbl) for im, lbl in zip(imgs, label)]
            imgs = [img_n_label[0] for img_n_label in img_n_label_cutmix]
            label = [img_n_label[1] for img_n_label in img_n_label_cutmix]

            self.prev_batch_for_cutmix = cur_batch_before_cutmix

        imgs = torch.stack(imgs)
        if type(label[0]) is int:
            label = torch.LongTensor(label)
        else:
            label = torch.stack(label)

        to_return = (imgs, label)
        if self.if_return_target_idx:
            to_return += (target_idx,)

        return to_return