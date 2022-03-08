import math

from ofa.imagenet_classification.run_manager import RunConfig
from ofa.utils import calc_learning_rate


class RunConfigMy(RunConfig):

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param, dataset, train_batch_size,
                 test_batch_size, valid_size, opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 mixup_alpha, model_init, validation_frequency, print_frequency, total_epochs):
        super().__init__(n_epochs, init_lr, lr_schedule_type, lr_schedule_param, dataset, train_batch_size,
                         test_batch_size, valid_size, opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                         mixup_alpha, model_init, validation_frequency, print_frequency)
        self.total_epochs = total_epochs

    def copy(self):
        return RunConfigMy(**self.config)

    """ learning rate """

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None, epoch_cumulative=None,
                             n_epochs_in_block_dynamic=None, n_epochs_in_block=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        if self.lr_schedule_type == 'cosine_nocycle':
            # cosine anneal to 0 over all the epochs
            t_total = self.total_epochs * nBatch
            t_cur = epoch_cumulative * nBatch + batch
            new_lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * t_cur / t_total))
        else:
            new_lr = calc_learning_rate(epoch, self.init_lr, self.n_epochs, batch, nBatch, self.lr_schedule_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def random_sub_train_loader(self, n_images, batch_size, img_size, num_worker=None, num_replicas=None, rank=None):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, img_size, num_worker, num_replicas, rank)
