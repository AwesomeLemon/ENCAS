import functools
import warnings
import os
import math
import numpy as np

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ofa.imagenet_classification.data_providers.imagenet import DataProvider

import utils
import utils_train
from dynamic_resolution_collator import DynamicResolutionCollator

from .auto_augment_tf import auto_augment_policy, AutoAugment

class ImagenetDataProvider(DataProvider):
    
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None, total_size=None, **kwargs):
        
        warnings.filterwarnings('ignore')
        self._save_path = save_path
        
        self.image_size = image_size
        self.distort_color = distort_color
        self.resize_scale = resize_scale
        self.if_flip = kwargs.get('if_flip', True)
        self.auto_augment = kwargs.get('auto_augment', 'v0')

        self.crop_pct = kwargs.get('crop_pct', None)
        self.if_timm = self.crop_pct is not None
        self.preproc_alphanet = kwargs.get('preproc_alphanet', False)

        self._train_transform_dict = {}
        self._valid_transform_dict = {}

        self.active_img_size = self.image_size
        valid_transforms = self.build_valid_transform()
        train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        self.train_dataset_actual = self.train_dataset(train_transforms)

        self.if_segmentation=False
        self.collator_train = DynamicResolutionCollator(1000)
        self.collator_val = DynamicResolutionCollator(1)
        self.collator_subtrain = DynamicResolutionCollator(1, if_return_target_idx=False)

        self.valid_dataset_actual = self.val_dataset(valid_transforms)
        self.train = train_loader_class(
            self.train_dataset_actual, batch_size=train_batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True, collate_fn=self.collator_train,
            worker_init_fn=utils_train.init_dataloader_worker_state, persistent_workers=True
        )
        self.valid = torch.utils.data.DataLoader(
            self.valid_dataset_actual, batch_size=test_batch_size,
            num_workers=n_worker
            , pin_memory=True, collate_fn=self.collator_val,
            persistent_workers=True
        )
        
        test_dataset = self.test_dataset(valid_transforms)
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker*2, pin_memory=True,
            collate_fn=self.collator_val, persistent_workers=True
        )

        self.collator_train.set_info_for_transforms(self.resize_class_lambda_train,
                                                    self.train_transforms_after_resize,
                                                    self.train_transforms_pre_resize)
        self.collator_val.set_info_for_transforms(self.resize_class_lambda_val,
                                                  self.val_transforms_after_resize, self.val_transforms_pre_resize)
        self.collator_subtrain.set_info_for_transforms(self.resize_class_lambda_train,
                                                       self.train_transforms_after_resize,
                                                       self.train_transforms_pre_resize)

    def set_collator_train_resolutions(self, resolutions):
        self.collator_train.set_resolutions(resolutions)
    
    @staticmethod
    def name():
        return 'imagenet'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return 1000
    
    @property
    def save_path(self):
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        dataset = datasets.ImageFolder(os.path.join(self.save_path, 'train_part'), _transforms)
        return dataset

    def val_dataset(self, _transforms):
        dataset = datasets.ImageFolder(os.path.join(self.save_path, 'imagenetv2_all'), _transforms)
        return dataset
    
    def test_dataset(self, _transforms):
        dataset = datasets.ImageFolder(os.path.join(self.save_path, 'val'), _transforms)
        return dataset
    
    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def create_resize_class_lambda_train(resize_transform_class, image_size, **kwargs):  # pickle can't handle lambdas
        return resize_transform_class((image_size, image_size), **kwargs)
    
    def build_train_transform(self, image_size=None, print_log=True):
        self.active_img_size = image_size
        default_image_size = 224
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, default_image_size))

        if self.active_img_size in self._train_transform_dict:
            return self._train_transform_dict[self.active_img_size]

        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        # OFA and AlphaNet preprocess ImageNet differently, and leads to significant performance changes
        # => I match the preprocessing to the supernetwork.
        #
        # In Alphanet, there are 2 resizes: when loading, images are always resized to 224*224;
        # and in the call to forward they are resized to the proper resolution of the current subnetwork.
        # I want to resize in advance, but still need to do the 2 resizes to match the preprocessing & performance
        # => for Alphanet, I add the first resize to 224 to the train_transforms, and I run all the transforms before
        #    the final resize.
        if not self.preproc_alphanet:
            if self.resize_scale:
                resize_scale = self.resize_scale
                self.resize_class_lambda_train = functools.partial(ImagenetDataProvider.create_resize_class_lambda_train,
                                                                   transforms.RandomResizedCrop, scale=[resize_scale, 1.0],
                                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            else:
                self.resize_class_lambda_train = functools.partial(utils_train.create_resize_class_lambda_train,
                                                                   transforms.Resize,
                                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            train_transforms = []
        else:
            resize_scale = self.resize_scale
            train_transforms = [transforms.RandomResizedCrop((default_image_size, default_image_size), scale=[resize_scale, 1.0],
                                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]

            self.resize_class_lambda_train = functools.partial(ImagenetDataProvider.resize_class_lambda_val, self.preproc_alphanet)

        if self.if_flip:
            train_transforms.append(transforms.RandomHorizontalFlip())

        if color_transform is not None:
            train_transforms.append(color_transform)

        if self.auto_augment:
            IMAGENET_PIXEL_MEAN = [123.675, 116.280, 103.530]
            aa_params = {
                "translate_const": int(default_image_size * 0.45),
                "img_mean": tuple(round(x) for x in IMAGENET_PIXEL_MEAN),
            }

            aa_policy = AutoAugment(auto_augment_policy(self.auto_augment, aa_params))
            train_transforms.append(aa_policy)

        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        if not self.preproc_alphanet:
            self.train_transforms_pre_resize = []
            self.train_transforms_after_resize = train_transforms
        else:
            self.train_transforms_pre_resize = train_transforms
            self.train_transforms_after_resize = []
        train_transforms = []

        # the transforms below are irrelevant (actual transforms will be in the collator)
        train_transforms = transforms.Compose(train_transforms)
        self._train_transform_dict[self.active_img_size] = train_transforms
        return train_transforms

    @staticmethod
    def resize_class_lambda_val(preproc_alphanet, image_size):
        if preproc_alphanet:
            return transforms.Resize((image_size, image_size),
                                     interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        return transforms.Compose([transforms.Resize(int(math.ceil(image_size / 0.875))),
                                                      transforms.CenterCrop(image_size)])

    def build_valid_transform(self, image_size=None):
        self.resize_class_lambda_val = functools.partial(ImagenetDataProvider.resize_class_lambda_val, self.preproc_alphanet)

        if not self.preproc_alphanet: # see the comment about "self.preproc_alphanet" in build_train_transform
            val_transforms = [
                transforms.ToTensor(),
                self.normalize
            ]
            self.val_transforms_pre_resize = []
            self.val_transforms_after_resize = val_transforms
        else:
            val_transforms = [
                transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize
            ]
            self.val_transforms_pre_resize = val_transforms
            self.val_transforms_after_resize = []

        val_transforms = []

        return transforms.Compose(val_transforms)

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    # Presumably, this also leads to OOM, but I haven't verified it on ImageNet, just assuming it's the same as with CIFARs.
    # def build_sub_train_loader(self, n_images, batch_size, img_size, num_worker=None, num_replicas=None, rank=None):
    #     # used for resetting running statistics
    #     if self.__dict__.get('sub_train_%d' % img_size, None) is None:
    #         if not hasattr(self, 'sub_data_loader'):
    #             if num_worker is None:
    #                 num_worker = self.train.num_workers
    #
    #             new_train_dataset = self.train_dataset(
    #                 self.build_train_transform(image_size=img_size, print_log=False))
    #
    #             g = torch.Generator()
    #             g.manual_seed(DataProvider.SUB_SEED)
    #
    #             # don't need to change sampling here (unlike in cifars) because val is not part of train
    #             n_samples = len(self.train.dataset.samples)
    #             rand_indexes = torch.randperm(n_samples, generator=g).tolist()
    #             chosen_indexes = rand_indexes[:n_images]
    #
    #             if num_replicas is not None:
    #                 sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
    #             else:
    #                 sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    #             self.collator_subtrain.set_resolutions([img_size])
    #             self.sub_data_loader = torch.utils.data.DataLoader(
    #                 new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
    #                 num_workers=0, pin_memory=False, collate_fn=self.collator_subtrain
    #             )
    #
    #         self.collator_subtrain.set_resolutions([img_size])
    #         self.__dict__['sub_train_%d' % img_size] = []
    #         for images, labels, *_ in self.sub_data_loader:
    #             self.__dict__['sub_train_%d' % img_size].append((images, labels))
    #     return self.__dict__['sub_train_%d' % img_size]

    def build_sub_train_loader(self, n_images, batch_size, img_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics of BN

        if not hasattr(self, 'sub_data_loader'):
            if num_worker is None:
                num_worker = self.train.num_workers

            new_train_dataset = self.train_dataset(self.build_train_transform(image_size=img_size, print_log=False))

            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)

            # don't need to change sampling here (unlike in cifars) because val is not part of train
            n_samples = len(self.train.dataset.samples)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            chosen_indexes = rand_indexes[:n_images]

            sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            self.collator_subtrain.set_resolutions([img_size])
            self.sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=False, collate_fn=self.collator_subtrain, persistent_workers=True
            )
        else:
            self.collator_subtrain.set_resolutions([img_size])

        return self.sub_data_loader