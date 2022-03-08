from data_providers.imagenet import *
from data_providers.cifar import CIFAR10DataProvider, CIFAR100DataProvider
from ofa.imagenet_classification.run_manager.run_config import RunConfig

from run_manager.run_config_my import RunConfigMy

class ImagenetRunConfig(RunConfig):
    def __init__(self, n_epochs=1, init_lr=1e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=128, test_batch_size=512, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 data_path='/mnt/datastore/ILSVRC2012',
                 **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.imagenet_data_path = data_path
        self.kwargs = kwargs

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.imagenet_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size, **self.kwargs
            )
        return self.__dict__['_data_provider']


class CIFARRunConfig(RunConfigMy):
    def __init__(self, n_epochs=5, init_lr=0.01, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar10', train_batch_size=96, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=2, resize_scale=0.08, distort_color=None, image_size=224,
                 data_path='/mnt/datastore/CIFAR',
                 **kwargs):
        super(CIFARRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency, kwargs['total_epochs']
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.cifar_data_path = data_path
        self.kwargs = kwargs

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == CIFAR10DataProvider.name():
                DataProviderClass = CIFAR10DataProvider
            elif self.dataset == CIFAR100DataProvider.name():
                DataProviderClass = CIFAR100DataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.cifar_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size, **self.kwargs
            )
        return self.__dict__['_data_provider']

def get_run_config(**kwargs):
    if 'init_lr' in kwargs:
        if kwargs['init_lr'] is None: # use dataset-specific init_lr by default
            del kwargs['init_lr']
    if kwargs['dataset'] == 'imagenet':
        run_config = ImagenetRunConfig(**kwargs)
    elif kwargs['dataset'].startswith('cifar'):
        run_config = CIFARRunConfig(**kwargs)
    else:
        raise NotImplementedError

    return run_config


