import functools
import math
import numpy as np

import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from ofa.imagenet_classification.data_providers.imagenet import DataProvider
from timm.data import rand_augment_transform

import utils_train
from utils import _pil_interp

from dynamic_resolution_collator import DynamicResolutionCollator
import utils
from utils_train import Cutout


class CIFARBaseDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=96, test_batch_size=256, valid_size=None,
                 n_worker=2, resize_scale=0.08, distort_color=None, image_size=224, num_replicas=None, rank=None,
                 total_size=None, **kwargs):

        self._save_path = save_path

        self.image_size = image_size
        self.distort_color = distort_color
        self.resize_scale = resize_scale
        self.cutout_size = kwargs['cutout_size']
        self.auto_augment = kwargs.get('auto_augment', 'rand-m9-mstd0.5')
        self.if_flip = kwargs.get('if_flip', True)
        self.if_center_crop = kwargs.get('if_center_crop', True)
        self.if_cutmix = kwargs.get('if_cutmix', False)

        self._valid_transform_dict = {}
        self._train_transform_dict = {}

        self.active_img_size = self.image_size
        valid_transforms = self.build_valid_transform()

        train_transforms = self.build_train_transform()
        self.train_dataset_actual = self.train_dataset(train_transforms)
        n_datapoints = len(self.train_dataset_actual.data)

        self.cutmix_kwargs = None
        if self.if_cutmix:
            self.cutmix_kwargs = {'beta': 1.0, 'prob': 0.5, 'n_classes': self.n_classes}
        # depending on combinations of flags may need even more than a 1000:
        self.collator_train = DynamicResolutionCollator(1000, if_cutmix=self.if_cutmix, cutmix_kwargs=self.cutmix_kwargs)
        self.collator_val = DynamicResolutionCollator(1)
        self.collator_subtrain = DynamicResolutionCollator(1, if_return_target_idx=False, if_cutmix=self.if_cutmix,
                                                           cutmix_kwargs=self.cutmix_kwargs)

        assert valid_size is not None
        if total_size is not None:
            n_datapoints = total_size
        if not isinstance(valid_size, int):
            assert isinstance(valid_size, float) and 0 < valid_size < 1
            valid_size = int(n_datapoints * valid_size)

        self.valid_dataset_actual = self.train_dataset(valid_transforms)
        train_indexes, valid_indexes = self.random_sample_valid_set(n_datapoints, valid_size)

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
        # for validation use Subset instead of SubsetRandomSampler to keep the ordering the same
        self.valid_dataset_actual = torch.utils.data.Subset(self.valid_dataset_actual, valid_indexes)

        self.train = torch.utils.data.DataLoader(
            self.train_dataset_actual, batch_size=train_batch_size, sampler=train_sampler,
            num_workers=n_worker, pin_memory=True, collate_fn=self.collator_train,
            worker_init_fn=utils_train.init_dataloader_worker_state, persistent_workers=True
        )
        self.valid = torch.utils.data.DataLoader(
            self.valid_dataset_actual, batch_size=test_batch_size, num_workers=n_worker, pin_memory=True,
            prefetch_factor=1, persistent_workers=True, collate_fn=self.collator_val
        )

        test_dataset = self.test_dataset(valid_transforms)
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
            collate_fn=self.collator_val, prefetch_factor=1
        )

        self.collator_train.set_info_for_transforms(self.resize_class_lambda_train,
                                                    self.train_transforms_after_resize)
        self.collator_val.set_info_for_transforms(self.resize_class_lambda_val,
                                                  self.val_transforms_after_resize)
        self.collator_subtrain.set_info_for_transforms(self.resize_class_lambda_train,
                                                       self.train_transforms_after_resize)

    def set_collator_train_resolutions(self, resolutions):
        self.collator_train.set_resolutions(resolutions)

    @staticmethod
    def name():
        raise NotImplementedError

    @property
    def n_classes(self):
        raise NotImplementedError

    def train_dataset(self, _transforms):
        raise NotImplementedError


    def test_dataset(self, _transforms):
        raise NotImplementedError

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def save_path(self):
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    @property
    def train_path(self):
        return self.save_path

    @property
    def valid_path(self):
        return self.save_path

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])

    def build_train_transform(self, image_size=None, print_log=True):
        self.active_img_size = image_size
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.active_img_size in self._train_transform_dict:
            return self._train_transform_dict[self.active_img_size]

        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if self.resize_scale:
            resize_scale = self.resize_scale
            self.resize_class_lambda_train = functools.partial(utils_train.create_resize_class_lambda_train,
                                                               transforms.RandomResizedCrop, scale=[resize_scale, 1.0],
                                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            self.resize_class_lambda_train = functools.partial(utils_train.create_resize_class_lambda_train,
                                                               transforms.Resize,
                                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        train_transforms = []
        if self.if_flip:
            train_transforms.append(transforms.RandomHorizontalFlip())
        if color_transform is not None:
            train_transforms.append(color_transform)

        if self.auto_augment:
            aa_params = dict(
                translate_const=int(image_size * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in [0.49139968, 0.48215827, 0.44653124]]),
            )
            aa_params['interpolation'] = _pil_interp('bicubic')
            train_transforms += [rand_augment_transform(self.auto_augment, aa_params)]

        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
            Cutout(length=self.cutout_size)
        ]

        self.train_transforms_after_resize = train_transforms
        train_transforms = []

        # these transforms are irrelevant
        train_transforms = transforms.Compose(train_transforms)
        self._train_transform_dict[self.active_img_size] = train_transforms
        return train_transforms

    @staticmethod
    def resize_class_lambda_val(if_center_crop, image_size):
        if if_center_crop:
            return transforms.Compose([
                transforms.Resize(int(math.ceil(image_size / 0.875)),
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size)])
        return transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    def build_valid_transform(self, image_size=None):
        self.resize_class_lambda_val = functools.partial(CIFAR100DataProvider.resize_class_lambda_val, self.if_center_crop)

        val_transforms = [transforms.ToTensor(),
                          self.normalize]
        self.val_transforms_after_resize = val_transforms
        val_transforms = []

        return transforms.Compose(val_transforms)

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(self, n_images, batch_size, img_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics of BN

        if not hasattr(self, 'sub_data_loader'):
            if num_worker is None:
                num_worker = self.train.num_workers

            new_train_dataset = self.train_dataset(self.build_train_transform(image_size=img_size, print_log=False))

            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)

            indices_train = self.train.sampler.indices
            n_indices = len(indices_train)
            rand_permutation = torch.randperm(n_indices, generator=g).numpy()
            indices_train = np.array(indices_train)[rand_permutation]
            chosen_indexes = indices_train[:n_images].tolist()

            sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            self.collator_subtrain.set_resolutions([img_size])
            self.sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=False, collate_fn=self.collator_subtrain, persistent_workers=True
            )
        else:
            self.collator_subtrain.set_resolutions([img_size])

        return self.sub_data_loader


class CIFAR10DataProvider(CIFARBaseDataProvider):
    @staticmethod
    def name():
        return 'cifar10'
    
    @property
    def n_classes(self):
        return 10
    
    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(root=self.train_path, train=True,
                                               download=True, transform=_transforms)
        return dataset
    
    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR10(root=self.valid_path, train=False,
                                               download=True, transform=_transforms)
        return dataset


class CIFAR100DataProvider(CIFARBaseDataProvider):
    @staticmethod
    def name():
        return 'cifar100'

    @property
    def n_classes(self):
        return 100

    def train_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(root=self.train_path, train=True,
                                               download=True, transform=_transforms)
        return dataset

    def test_dataset(self, _transforms):
        dataset = torchvision.datasets.CIFAR100(root=self.valid_path, train=False,
                                               download=True, transform=_transforms)
        return dataset
