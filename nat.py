import itertools
import os
import time
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from torch.cuda.amp import GradScaler

from ofa.utils import AverageMeter, accuracy
from tqdm import tqdm
from matplotlib import pyplot as plt

import utils
from networks.ofa_mbv3_my import OFAMobileNetV3My
from run_manager import get_run_config
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics

from networks.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from networks.proxyless_my import OFAProxylessNASNetsMy
from utils import validate_config, get_net_info

from searcher_wrappers.mo_gomea_wrapper import MoGomeaWrapper
from searcher_wrappers.nsga3_wrapper import Nsga3Wrapper
from searcher_wrappers.random_search_wrapper import RandomSearchWrapper
import subset_selectors
import gc
from filelock import FileLock
import dill
from utils_train import CutMixCrossEntropyLoss, LabelSmoothing

os.environ['MKL_THREADING_LAYER'] = 'GNU'
import json
import shutil
import numpy as np
from utils import get_correlation, alphabet_dict, get_metric_complement, setup_logging

from search_space import OFASearchSpace
from search_space.ensemble_ss import EnsembleSearchSpace
from acc_predictor.factory import get_acc_predictor

from pymoo.visualization.scatter import Scatter
plt.rcParams.update({'font.size': 16})

from collections import defaultdict

from utils import set_seed
import re
import yaml

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

class NAT:
    def __init__(self, kwargs):
        kwargs_copy = dict(kwargs)
        plt.rcParams['axes.grid'] = True
        def get_from_kwargs_or_default_kwargs(key_name):
            return kwargs.pop(key_name, default_kwargs[key_name][0])

        self.random_seed = kwargs.pop('random_seed', default_kwargs['random_seed'][0])
        set_seed(self.random_seed)

        # 1. search space & alphabets
        self.search_space_name = kwargs.pop('search_space', default_kwargs['search_space'][0])
        search_goal = kwargs.pop('search_goal', default_kwargs['search_goal'][0])
        self.if_cascade = search_goal == 'cascade'
        self.ensemble_ss_names = kwargs.pop('ensemble_ss_names', default_kwargs['ensemble_ss_names'][0])
        alphabet_names = kwargs.pop('alphabet', default_kwargs['alphabet'][0])
        alphabet_paths = [alphabet_dict[alphabet_name] for alphabet_name in alphabet_names]
        if self.search_space_name == 'ensemble':
            self.search_space = EnsembleSearchSpace(self.ensemble_ss_names,
                                                    [{'alphabet': alphabet_name, 'ensemble_size': len(alphabet_names)}
                                                     for alphabet_name in alphabet_names])
            self.alphabets = []
            self.alphabets_lb = []
            for alphabet_path in alphabet_paths:
                with open(alphabet_path, 'r') as f:
                    self.alphabets.append(list(map(int, f.readline().split(' '))))
                with open(alphabet_path.replace('.txt', '_lb.txt'), 'r') as f:
                    self.alphabets_lb.append(list(map(int, f.readline().split(' ')))) #lower bound
            # combined alphabets
            self.alphabet = list(itertools.chain(*self.alphabets))
            self.alphabet_lb = list(itertools.chain(*self.alphabets_lb))
        elif self.search_space_name == 'reproduce_nat':
            assert len(alphabet_names) == 2
            assert alphabet_names[0] == alphabet_names[1]
            alphabet_path = alphabet_paths[0]
            alphabet_name = alphabet_names[0]
            self.search_space = OFASearchSpace(alphabet=alphabet_name)
            with open(alphabet_path, 'r') as f:
                self.alphabet = list(map(int, f.readline().split(' ')))
            with open(alphabet_path.replace('.txt', '_lb.txt'), 'r') as f:
                self.alphabet_lb = list(map(int, f.readline().split(' ')))  # lower bound

        # 2. save & log
        self.path_logs = kwargs.pop('path_logs', default_kwargs['path_logs'][0])
        self.resume = kwargs.pop('resume', default_kwargs['resume'][0])
        if self.resume is not None:
            self.resume = os.path.join(self.path_logs, self.resume)
        save_name = kwargs.pop('experiment_name', default_kwargs['experiment_name'][0])
        self.path_logs = os.path.join(self.path_logs, save_name)
        Path(self.path_logs).mkdir(exist_ok=True)
        self.log_file_path = os.path.join(self.path_logs, '_log.txt')
        setup_logging(self.log_file_path)
        print(f'{self.path_logs=}')

        # 3. copy pre-trained supernets
        supernet_paths = kwargs.pop('supernet_path', default_kwargs['supernet_path'][0])
        print(f'{supernet_paths=}')
        supernet_paths_true = []
        for supernet_path in supernet_paths:
            # try:
            shutil.copy(supernet_path, self.path_logs)
            # except:
            #     pass
            supernet_paths_true.append(os.path.join(self.path_logs, os.path.basename(supernet_path)))
        self.supernet_paths = supernet_paths_true

        # 4. data
        trn_batch_size = get_from_kwargs_or_default_kwargs('trn_batch_size')
        vld_batch_size = get_from_kwargs_or_default_kwargs('vld_batch_size')
        n_workers = get_from_kwargs_or_default_kwargs('n_workers')
        vld_size = get_from_kwargs_or_default_kwargs('vld_size')
        total_size = get_from_kwargs_or_default_kwargs('total_size')
        data_path = get_from_kwargs_or_default_kwargs('data')
        init_lr = get_from_kwargs_or_default_kwargs('init_lr')
        lr_schedule_type = kwargs.pop('lr_schedule_type', default_kwargs['lr_schedule_type'][0])
        cutout_size = kwargs.pop('cutout_size', default_kwargs['cutout_size'][0])
        weight_decay = kwargs.pop('weight_decay', default_kwargs['weight_decay'][0])
        if_center_crop = kwargs.pop('if_center_crop', default_kwargs['if_center_crop'][0])
        auto_augment = kwargs.pop('auto_augment', default_kwargs['auto_augment'][0])
        resize_scale = kwargs.pop('resize_scale', default_kwargs['resize_scale'][0])
        if_cutmix = kwargs.pop('if_cutmix', default_kwargs['if_cutmix'][0])

        self.iterations = kwargs.pop('iterations', default_kwargs['iterations'][0])
        self.dataset = kwargs.pop('dataset', default_kwargs['dataset'][0])
        self.n_epochs = kwargs.pop('n_epochs', default_kwargs['n_epochs'][0])
        # in order not to pickle "self", create variables without it:
        dataset, n_epochs, iterations, ensemble_ss_names = self.dataset, self.n_epochs, self.iterations, self.ensemble_ss_names
        self.run_config_lambda = lambda: get_run_config(
            dataset=dataset, data_path=data_path, image_size=256,
            n_epochs=n_epochs, train_batch_size=trn_batch_size, test_batch_size=vld_batch_size,
            n_worker=n_workers, valid_size=vld_size, total_size=total_size, dataset_name=dataset,
            total_epochs=(iterations + 1) * n_epochs, lr_schedule_type=lr_schedule_type,
            weight_decay=weight_decay, init_lr=init_lr, cutout_size=cutout_size, if_center_crop=if_center_crop,
            auto_augment=auto_augment, resize_scale=resize_scale, if_cutmix=if_cutmix,
            preproc_alphanet='alphanet' in ensemble_ss_names # needed only for imagenet
        )

        # 5. search algorithm
        run_config = self.run_config_lambda() # need to create run_config here just to get the number of classes
        self.n_classes = run_config.data_provider.n_classes
        gomea_exe_path = get_from_kwargs_or_default_kwargs('gomea_exe')
        search_algo = kwargs.pop('search_algo', default_kwargs['search_algo'][0])
        assert search_algo in ['nsga3', 'mo-gomea', 'random']
        search_algo_class = {'nsga3': Nsga3Wrapper, 'mo-gomea': MoGomeaWrapper,
                             'random': RandomSearchWrapper}[search_algo]
        init_with_nd_front_size = kwargs.pop('init_with_nd_front_size', default_kwargs['init_with_nd_front_size'][0])
        n_surrogate_evals = kwargs.pop('n_surrogate_evals', default_kwargs['n_surrogate_evals'][0])
        self.sec_obj = kwargs.pop('sec_obj', default_kwargs['sec_obj'][0])
        self.if_add_archive_to_candidates = get_from_kwargs_or_default_kwargs('add_archive_to_candidates')
        self.search_wrapper = search_algo_class(self.search_space, self.sec_obj, self.path_logs,
                                                self.n_classes, self.supernet_paths,
                                                n_surrogate_evals, self.if_add_archive_to_candidates,
                                                alphabet=self.alphabet, alphabet_path=alphabet_paths, alphabet_name=alphabet_names,
                                                init_with_nd_front_size=init_with_nd_front_size, gomea_exe_path=gomea_exe_path,
                                                n_image_channels=3,
                                                dataset=self.dataset, search_space_name=self.search_space_name,
                                                alphabet_lb=self.alphabet_lb, ensemble_ss_names=self.ensemble_ss_names)
        subset_selector_name = kwargs.pop('subset_selector', default_kwargs['subset_selector'][0])
        archive_size = kwargs.pop('n_iter', default_kwargs['n_iter'][0])
        self.subset_selector = subset_selectors.create_subset_selector(subset_selector_name, archive_size)

        # 6. create lambdas for creating supernets (engines)
        # Why lambdas? Because they can be used multiple times and in subprocesses to create engines
        # with the same setup (but loaded weights will be different because I'll be overwriting save files)
        self.create_engine_lambdas = []
        ss_name_to_class = {'alphanet': AttentiveNasDynamicModel, 'ofa': OFAMobileNetV3My,
                            'proxyless': OFAProxylessNASNetsMy}
        use_gradient_checkpointing = get_from_kwargs_or_default_kwargs('use_gradient_checkpointing')
        for ss_name in self.ensemble_ss_names:
            class_to_use = ss_name_to_class[ss_name]
            self.create_engine_lambdas.append(NAT.make_lambda_for_engine_creation(class_to_use, self.n_classes,
                                                                                  use_gradient_checkpointing,
                                                                                  self.dataset, ss_name))

        # 7. loss functions
        label_smoothing = kwargs.pop('label_smoothing', default_kwargs['label_smoothing'][0])
        if label_smoothing == 0.0:
            if if_cutmix:
                self.train_criterion = CutMixCrossEntropyLoss()
            else:
                self.train_criterion = torch.nn.CrossEntropyLoss()
            self.val_criterion = torch.nn.CrossEntropyLoss()
        else:
            assert not if_cutmix
            print(f'Using label smoothing with coefficient == {label_smoothing}')
            self.train_criterion = LabelSmoothing(label_smoothing)
            self.val_criterion = LabelSmoothing(label_smoothing)

        # 8. used later
        self.initial_sample_size = kwargs.pop('n_doe', default_kwargs['n_doe'][0])
        self.predictor = kwargs.pop('predictor', default_kwargs['predictor'][0])
        self.n_warmup_epochs = kwargs.pop('n_warmup_epochs', default_kwargs['n_warmup_epochs'][0])
        self.if_amp = get_from_kwargs_or_default_kwargs('if_amp')
        self.rbf_ensemble_size = kwargs.pop('rbf_ensemble_size', default_kwargs['rbf_ensemble_size'][0])
        self.if_check_duplicates = not kwargs.pop('dont_check_duplicates', default_kwargs['dont_check_duplicates'][0])
        self.if_sample_configs_to_train = get_from_kwargs_or_default_kwargs('sample_configs_to_train')
        self.store_checkpoint_freq = kwargs.pop('store_checkpoint_freq', default_kwargs['store_checkpoint_freq'][0])
        self.get_scalar_from_accuracy = lambda acc: acc[0].item()
        self.lock = FileLock(os.path.join(str(Path(self.path_logs).parents[1]),
                              f'gpu_{os.environ["CUDA_VISIBLE_DEVICES"].replace(",", "_")}.lock'))

        # 9. save config
        with open(os.path.join(self.path_logs, 'config_msunas.yml'), 'w') as f:
            yaml.dump(kwargs_copy, f)

    def search(self):
        worst_top1_err, worst_flops = 40, 4000
        ref_pt = np.array([worst_top1_err, worst_flops])
        archive, first_iteration = self.create_or_restore_archive(ref_pt)

        for it in range(first_iteration, self.iterations + 1):
            archive, *_ = self.search_step(archive, it, ref_pt)

    def search_step(self, archive, it, ref_pt):
        acc_predictor, pred_for_archive = self.fit_surrogate(archive, self.alphabet, self.alphabet_lb)
        candidates, pred_for_candidates = self.surrogate_search(archive, acc_predictor, it=it)
        objs_evaluated = self.train_and_evaluate(candidates, it)
        candidates_top1_err, candidates_complexity = objs_evaluated[0], objs_evaluated[1]
        # correlation for accuracy
        rmse, rho, tau = get_correlation(np.hstack((pred_for_archive[:, 0], pred_for_candidates[:, 0])),
                                         np.array([x[1] for x in archive] + candidates_top1_err))
        # correlation for flops
        if self.if_cascade:
            _, rho_flops, _ = get_correlation(np.hstack((pred_for_archive[:, 1], pred_for_candidates[:, 1])),
                                              np.array([x[2] for x in archive] + candidates_complexity))
            print(f'{rho_flops=}')

        candidates_with_objs = []
        for member in zip(candidates, *objs_evaluated):
            candidates_with_objs.append(member)
        if self.if_add_archive_to_candidates:
            archive = candidates_with_objs  # because archive was added to candidates in self.surrogate_search
        else:
            archive += candidates_with_objs  # because candidates don't include archive

        hv = utils.compute_hypervolume(ref_pt, np.column_stack(list(zip(*archive))[1:3]))
        hv_candidates = utils.compute_hypervolume(ref_pt, np.column_stack(list(zip(*candidates_with_objs))[1:3]))

        print(f'\nIter {it}: hv = {hv:.2f}')
        print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall’s Tau = {tau:.4f}")
        with open(os.path.join(self.path_logs, 'iter_{}.stats'.format(it)), 'w') as handle:
            json.dump({'archive': archive, 'candidates': candidates_with_objs, 'hv': hv, 'hv_candidates': hv_candidates,
                       'surrogate': {'model': self.predictor, 'name': acc_predictor.name, 'winner': acc_predictor.name,
                           'rmse': rmse, 'rho': rho, 'tau': tau}}, handle)

        self.plot_archive(archive, candidates_top1_err, candidates, candidates_complexity, it, pred_for_candidates)
        return archive, {'acc_val_max': get_metric_complement(np.min([x[1] for x in archive]))}

    def create_or_restore_archive(self, ref_pt):
        if self.resume:
            # loads the full archive, not just the candidates of the latest iteration
            data = json.load(open(self.resume))
            iter = re.search('(\d+)(?!.*\d)', self.resume)[0]  # last number in the name
            archive, first_iteration = data['archive'], int(iter)
            if first_iteration == 0:
                # MO-GOMEA needs the archive of previous iteration => copy it into the folder of the current run
                try:
                    shutil.copy(self.resume, self.path_logs)
                except shutil.SameFileError:
                    pass
            first_iteration += 1

        else:
            archive = []
            arch_doe = self.search_space.initialize(self.initial_sample_size)

            if self.n_warmup_epochs > 0:
                print(f'Warmup: train for {self.n_warmup_epochs} epochs')
                self.lock.acquire()
                st = time.time()
                self._train(arch_doe, -1, n_epochs=self.n_warmup_epochs, if_warmup=True)
                ed = time.time()
                print(f'Train time = {ed - st}')
                self.lock.release()

            objs_evaluated = self.train_and_evaluate(arch_doe, 0)

            for member in zip(arch_doe, *objs_evaluated):
                archive.append(member)

            hv = utils.compute_hypervolume(ref_pt, np.column_stack(list(zip(*archive))[1:3]))

            with open(os.path.join(self.path_logs, 'iter_0.stats'), 'w') as handle:
                json.dump({'archive': archive, 'candidates': [], 'hv': hv, 'hv_candidates': hv,
                           'surrogate': {}}, handle)
            first_iteration = 1
        return archive, first_iteration

    def fit_surrogate(self, archive, alphabet, alphabet_lb):
        if 'rbf_ensemble_per_ensemble_member' not in self.predictor:
            inputs = np.array([self.search_space.encode(x[0]) for x in archive])
            targets = np.array([x[1] for x in archive])
            print(len(inputs), len(inputs[0]))
            assert len(inputs) > len(inputs[0]), '# of training samples have to be > # of dimensions'
            inputs_additional = {}
        else:
            inputs = list(zip(*[self.search_space.encode(x[0], if_return_separate=True) for x in archive]))
            inputs = [np.array(i) for i in inputs]
            targets = {}

            metric_per_member = list(zip(*[x[-1][0] for x in archive]))
            targets['metrics_sep'] = [np.array(x) for x in metric_per_member]

            targets['flops_cascade'] = np.array([x[2] for x in archive])

            inputs_additional = {}
            flops_per_member = list(zip(*[x[-1][1] for x in archive]))
            flops_per_member = [np.array(x) for x in flops_per_member]
            flops_per_member = np.array(flops_per_member, dtype=np.int).T

            inputs_for_flops = [i[:, -2:] for i in inputs]
            inputs_for_flops = np.concatenate(inputs_for_flops, axis=1)
            inputs_for_flops = np.hstack((inputs_for_flops, flops_per_member)) # n_samples, (ensemble_size*3) // because positions, thresholds, flops for each member

            inputs_additional['inputs_for_flops'] = inputs_for_flops

            inputs_for_flops_alphabet = np.concatenate([a[-2:] for a in self.alphabets] + [[2000] * len(self.alphabets)]) # for flops: they shouldn't be bigger than 2000
            inputs_for_flops_alphabet_lb = np.concatenate([a[-2:] for a in self.alphabets_lb] + [[0] * len(self.alphabets)])

            inputs_additional['inputs_for_flops_alphabet'] = inputs_for_flops_alphabet
            inputs_additional['inputs_for_flops_alphabet_lb'] = inputs_for_flops_alphabet_lb

            print(len(inputs), len(inputs[0]), len(inputs[0][0]))
            assert len(inputs[0]) > max([len(x) for x in inputs[0]]), '# of training samples have to be > # of dimensions'

            if 'combo' in self.predictor:
                targets['metrics_ens'] = np.array([x[1] for x in archive])

        if self.search_space_name == 'reproduce_nat':
            # NAT uses only 100 out of 300 archs to fit the predictor
            # we can use the same subset selector, but need to change number of archs to select, and then change it back
            normal_n_select = self.subset_selector.n_select
            self.subset_selector.n_select = 100
            errs = 100 - targets # reference selection assumes minimization
            flops = np.array([x[2] for x in archive])
            objs = np.vstack((errs, flops)).T
            # ReferenceBasedSelector doesn't actually use archive
            indices = self.subset_selector.select([], objs)
            self.subset_selector.n_select = normal_n_select
            actual_inputs_for_fit = inputs[indices]
            targets = targets[indices]
            print(f'{actual_inputs_for_fit.shape=}, {targets.shape=}')
        else:
            actual_inputs_for_fit = inputs

        acc_predictor = get_acc_predictor(self.predictor, actual_inputs_for_fit, targets, np.array(alphabet),
                                          np.array(alphabet_lb), inputs_additional=inputs_additional,
                                          ensemble_size=self.rbf_ensemble_size)
        if 'rbf_ensemble_per_ensemble_member' in self.predictor:
            inputs = np.concatenate(inputs, axis=1) # for creating predictor need them separately, but for prediction need a single vector
            inputs = {'for_acc': inputs, 'for_flops': inputs_for_flops}
        # to calculate predictor correlation:
        predictions = acc_predictor.predict(inputs)
        return acc_predictor, predictions

    def surrogate_search(self, archive, predictor, it=0):
        seed_cur = self.random_seed + it
        set_seed(seed_cur)

        st = time.time()
        genomes, objs = self.search_wrapper.search(archive, predictor, it, seed=seed_cur)
        ed = time.time()
        print(f'Search time = {ed - st}')

        if self.if_check_duplicates:
            archive_genomes = [x[0] for x in archive]
            new_genomes_decoded = [self.search_space.decode(x) for x in genomes]
            not_duplicate = np.logical_not([x in archive_genomes for x in new_genomes_decoded])
        else:
            not_duplicate = np.full(genomes.shape[0], True, dtype=bool)

        st = time.time()
        indices = self.subset_selector.select(archive, objs[not_duplicate])
        genomes_selected = genomes[not_duplicate][indices]
        objs_selected = objs[not_duplicate][indices]
        ed = time.time()
        print(f'Select time = {ed - st}')

        genomes_selected, unique_idx = np.unique(genomes_selected, axis=0, return_index=True)
        objs_selected = objs_selected[unique_idx]

        candidates = [self.search_space.decode(x) for x in genomes_selected]

        return candidates, objs_selected

    def train_and_evaluate(self, archs, it, n_epochs=None, if_warmup=False):
        self.lock.acquire()

        st = time.time()
        self._train(archs, it, n_epochs=n_epochs, if_warmup=if_warmup)
        ed = time.time()
        print(f'Train time = {ed - st}')

        self.lock.release()

        st = time.time()
        eval_res = self._evaluate_model_list(archs)
        ed = time.time()
        print(f'Eval time = {ed - st}')

        gc.collect()
        torch.cuda.empty_cache()

        # self.lock.release()
        return eval_res

    @staticmethod
    def _init_subprocess(log_file_path, fraction):
        setup_logging(log_file_path)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

    def _train(self, archs, it, number_to_add_to_i=0, n_epochs=None, if_warmup=False):
        thread_pool = ProcessPoolExecutor(max_workers=1,
                                          initializer=NAT._init_subprocess, initargs=(self.log_file_path, 0.44,))
                                          # initializer=setup_logging, initargs=(self.log_file_path,))
        n_engines_to_train = len(self.create_engine_lambdas)
        if self.search_space_name == 'ensemble':
            percent_train_per_engine = [1 / n_engines_to_train] * len(self.ensemble_ss_names)
            lambda_select_archs_per_engine = [lambda _: True] * len(self.ensemble_ss_names)
        elif self.search_space_name == 'reproduce_nat':
            n_archs_w1_0 = np.sum([config['w'] == 1.0 for config in archs])
            percent_w1_0 = n_archs_w1_0 / len(archs)
            print(f'{percent_w1_0=}')
            percent_train_per_engine = [percent_w1_0, 1 - percent_w1_0]
            lambda_select_archs_per_engine = [lambda arch: arch['w'] == 1.0, lambda arch: arch['w'] == 1.2]
        for i, (ss_name, create_engine_lambda) in enumerate(zip(self.ensemble_ss_names, self.create_engine_lambdas)):
            dump_path_train1 = os.path.join(self.path_logs, 'dump_train1.pkl')
            if self.search_space_name == 'ensemble':
                archs_cur = [arch[i] for arch in archs]  # archs is a list of lists, each of which contains configs for an ensemble
                search_space = self.search_space.search_spaces[i]
            elif self.search_space_name == 'reproduce_nat':
                archs_cur = archs
                search_space = self.search_space

            actual_logs_path = self.path_logs
            with open(dump_path_train1, 'wb') as f:
                dill.dump((archs_cur, it, number_to_add_to_i, n_epochs, if_warmup, create_engine_lambda,
                           self.random_seed + i, self.run_config_lambda, self.if_sample_configs_to_train,
                           search_space, self.dataset, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                           self.train_criterion, self.get_scalar_from_accuracy, actual_logs_path,
                           self.supernet_paths[i], lambda_select_archs_per_engine[i], percent_train_per_engine[i],
                           self.store_checkpoint_freq, self.sec_obj, self.if_amp), f)

            future = thread_pool.submit(NAT._train_one_supernetwork_stateless, dump_path_train1)
            future.result()

        del thread_pool
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _train_one_supernetwork_stateless(args_dump_path):
        with open(args_dump_path, 'rb') as f:
            archs, it, number_to_add_to_i, n_epochs, if_warmup, create_engine_lambda, random_seed, run_config_lambda, \
            if_sample_configs_to_train, search_space, dataset_name, device, train_criterion, \
            get_scalar_from_accuracy, path_logs, supernet_path, lambda_filter_archs, percent_steps_to_take, \
            store_checkpoint_freq, sec_obj, if_amp \
                = dill.load(f)
        set_seed(random_seed + it) # need to keep changing the seed, otherwise all the epochs use the same random values
        run_config = run_config_lambda()
        engine, optimizer = create_engine_lambda(supernet_path, run_config, device=device)

        n_batches = len(run_config.train_loader)
        if n_epochs is None:
            n_epochs = run_config.n_epochs

        if if_sample_configs_to_train:
            configs_encoded = np.array([search_space.encode(c) for c in archs])
            unique_with_counts = [np.unique(i, return_counts=True) for i in configs_encoded.T]
            unique_with_probs = [(u, c / configs_encoded.shape[0]) for (u, c) in unique_with_counts]
            sample = np.array([np.random.choice(u, n_epochs * n_batches, p=p)
                               for (u, p) in unique_with_probs])
            sample_decoded = [search_space.decode(c) for c in sample.T]
        else:
            archs = [arch for arch in archs if lambda_filter_archs(arch)]

        all_resolutions = [arch['r'] for arch in archs]
        run_config.data_provider.collator_train.set_resolutions(all_resolutions)

        n_steps_to_take = int(n_epochs * n_batches * percent_steps_to_take)
        n_epochs_to_take = n_steps_to_take // n_batches

        if if_amp:
            scaler = GradScaler()

        step = 0
        epoch = 0  # for saving not to fail when n_epochs == 0
        for epoch in range(0, n_epochs):
            if step == n_steps_to_take: #don't waste time initializing dataloader threads for the epochs that won't run
                break
            engine.train()

            losses = AverageMeter()
            metric_dict = defaultdict(lambda: AverageMeter())
            data_time = AverageMeter()

            with tqdm(total=n_batches,
                      desc='{} Train #{}'.format(run_config.dataset, epoch + number_to_add_to_i), ncols=175) as t:
                end = time.time()
                for i, (images, labels, config_idx) in enumerate(run_config.train_loader):
                    time_diff = time.time() - end
                    data_time.update(time_diff)

                    if step == n_steps_to_take:
                        break
                    step += 1

                    if if_sample_configs_to_train:
                        config = sample_decoded[epoch * n_batches + i]  # all the variables other than resolution have already been sampled in advance
                    else:
                        config = archs[config_idx]
                    if search_space.name in ['ofa', 'proxyless']:
                        config = validate_config(config)
                    engine.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'], w=config['w'])
                    if if_warmup:
                        # new_lr = run_config.init_lr
                        # previously warmup had constant lr, switch to linear warmup
                        new_lr = (step / n_steps_to_take) * run_config.init_lr
                    else:
                        new_lr = run_config.adjust_learning_rate(optimizer, epoch, i, n_batches,
                                                                 it * n_epochs + epoch, n_epochs_to_take, n_epochs)

                    images, labels = images.to(device), labels.to(device)
                    if not if_amp:
                        output = engine(images)
                        loss = train_criterion(output, labels)
                    else:
                        with torch.cuda.amp.autocast():
                            output = engine(images)
                            loss = train_criterion(output, labels)

                    optimizer.zero_grad()
                    if not if_amp:
                        loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    losses.update(loss.item(), images.size(0))
                    labels_for_acc = labels
                    if len(labels.shape) > 1:
                        labels_for_acc = torch.argmax(labels, dim=-1)
                    acc1 = accuracy(output, labels_for_acc, topk=(1,))
                    acc1 = get_scalar_from_accuracy(acc1)
                    metric_dict['top1'].update(acc1, output.size(0))

                    t.set_postfix({'loss': losses.avg,
                        **{key: metric_dict[key].avg for key in metric_dict},
                        'img_size': images.size(2),
                        'lr': new_lr,
                        'data_time': data_time.avg})
                    t.update(1)
                    end = time.time()

        width_mult = engine.width_mult[0]
        # save the new supernet weights
        save_path_iter = os.path.join(path_logs, f'iter_{it}')
        Path(save_path_iter).mkdir(exist_ok=True)
        def save_engine_weights(save_path):
            dict_to_save = {'epoch': epoch,
                            'model_state_dict': engine.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'width_mult': width_mult}

            engine.state_dict(torch.save(dict_to_save, save_path))
        if (it + 1) % store_checkpoint_freq == 0:
            save_engine_weights(os.path.join(save_path_iter, os.path.basename(supernet_path)))
        # but additionally always save in the main log folder: needed for the whole thing to keep on working
        # ("train" updates & overwrites these weights, "eval" uses the latest version of the weights)
        save_engine_weights(supernet_path)

    def _evaluate_model_list(self, archs, number_to_add_to_i=0):
        engines = []
        for i, (create_engine_lambda, supernet_path) in enumerate(zip(self.create_engine_lambdas, self.supernet_paths)):
            run_config = self.run_config_lambda() # only used within create_engine_lambda
            engine, opt = create_engine_lambda(supernet_path, run_config, to_cuda=False)
            engines.append(engine)
        def capture_variable_in_lambda(t):
            return lambda _: t
        get_engines = [capture_variable_in_lambda(engine) for engine in engines]

        thread_pool = ProcessPoolExecutor(max_workers=1,
                                          # initializer=setup_logging, initargs=(self.log_file_path,))
                                          initializer=NAT._init_subprocess, initargs=(self.log_file_path,0.44))

        dump1_path = os.path.join(self.path_logs, 'dump1.pkl')
        with open(dump1_path, 'wb') as f:
            dill.dump({'archs': archs, 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                       'val_criterion': self.val_criterion, 'get_scalar_from_accuracy': self.get_scalar_from_accuracy,
                       'sec_obj': self.sec_obj, 'search_space_ensemble': self.search_space,
                       'get_engines': get_engines, 
                       'run_config_lambda': self.run_config_lambda, 'number_to_add_to_i': number_to_add_to_i,
                       'if_ensemble_perf_per_member': 'rbf_ensemble_per_ensemble_member' in self.predictor,
                       'if_cascade': self.if_cascade}, f)
        future = thread_pool.submit(NAT._evaluate_model_list_stateless, dump1_path)
        res = future.result()
        del thread_pool

        try:
            os.remove(dump1_path)
        except:
            pass

        return tuple(res)


    @staticmethod
    def _evaluate_model_list_stateless(args_dump_path): # must be called by _evaluate_model_list
        with open(args_dump_path, 'rb') as f:
            kwargs_loaded = dill.load(f)
        top1_errs = []
        complexities = []
        if_ensemble_perf_per_member = kwargs_loaded['if_ensemble_perf_per_member']
        if if_ensemble_perf_per_member:
            perf_and_flops_per_subnet_all = []

        run_config = kwargs_loaded['run_config_lambda']()
        kwargs_loaded['run_config'] = run_config
        archs = kwargs_loaded['archs']
        for i_config in range(len(archs)):
            kwargs_loaded['config_ensemble'] = archs[i_config]
            kwargs_loaded['i_config'] = i_config
            top1_err, complexity, perf_and_flops_per_subnet = NAT._evaluate_model(**kwargs_loaded)
            top1_errs.append(top1_err)
            complexities.append(complexity)
            if if_ensemble_perf_per_member:
                perf_and_flops_per_subnet_all.append(perf_and_flops_per_subnet)

        to_return = top1_errs, complexities
        if if_ensemble_perf_per_member:
            to_return += (perf_and_flops_per_subnet_all,)
        return to_return

    @staticmethod
    def _evaluate_model(device, val_criterion,
                        get_scalar_from_accuracy, sec_obj, search_space_ensemble, get_engines,
                        run_config, config_ensemble, i_config, number_to_add_to_i,  if_ensemble_perf_per_member,
                        if_cascade, **kwargs): #don't need kwargs, have them to ignore irrelevant parameters passed here
        print('started _evaluate_model')
        subnets = []
        resolution_max = -1
        resolutions_list = []
        thresholds = None
        if if_cascade:
            positions_list = []
            thresholds = []

        if type(search_space_ensemble) is OFASearchSpace: # reproduce_nat
            search_spaces = [search_space_ensemble]
            config_ensemble = [config_ensemble]
            # I had a bug caused by the fact that the zero-th engine is used every time
            if config_ensemble[0]['w'] == 1.0:
                get_engines = [get_engines[0]]
            else:
                get_engines = [get_engines[1]]
        else:
            search_spaces = search_space_ensemble.search_spaces

        vld_batch_size = run_config.valid_loader.batch_size
        for i, search_space in enumerate(search_spaces):
            if search_space.name in ['ofa', 'proxyless']:
                config_ensemble[i].update(validate_config(config_ensemble[i])) # tuple doesn't support item assignment
            resolution, subnet = NAT._extract_subnet_from_supernet(config_ensemble[i], get_engines[i], run_config, vld_batch_size, device)
            subnets.append(subnet)

            resolution_max = max(resolution_max, resolution)
            resolutions_list.append(resolution)
            if if_cascade:
                positions_list.append(config_ensemble[i]['position'])
                thresholds.append(config_ensemble[i]['threshold'])
        if if_cascade:
            idx = np.argsort(positions_list)[::-1]
            thresholds = np.array(thresholds)[idx].tolist()
            resolutions_list = np.array(resolutions_list)[idx].tolist()
            subnets = np.array(subnets)[idx].tolist()
            reverse_idx = np.argsort(idx) #https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        resolution = resolution_max
        run_config.valid_loader.collate_fn.set_resolutions([resolution])  # at this point all resolutions should be the same

        metric_dict_val = defaultdict(lambda: AverageMeter())
        losses_val = AverageMeter()
        n_input_channels = -1

        if if_cascade:
            n_not_predicted_per_stage = [0 for _ in range(len(subnets) - 1)]

        with torch.no_grad(), torch.cuda.amp.autocast():
            with tqdm(total=len(run_config.valid_loader),
                      desc='{} Val #{}'.format(run_config.dataset, i_config + number_to_add_to_i),
                      ncols=200) as t:
                # print(i_cuda, 'before dataloader_val loop')
                for i, (images, labels, *other_stuff) in enumerate(run_config.valid_loader):
                    images, labels = images.to(device), labels.to(device)
                    images_orig = None # don't make a backup unless I need to
                    output = None
                    if if_cascade:
                        idx_more_predictions_needed = torch.ones(images.shape[0], dtype=torch.bool)

                    for i_subnet, subnet in enumerate(subnets):
                        if i_subnet > 0:
                            cur_threshold = thresholds[i_subnet - 1]
                            idx_more_predictions_needed[torch.max(output, dim=1).values >= cur_threshold] = False
                            output_tmp = output[idx_more_predictions_needed]
                            if len(output_tmp) == 0:
                                n_not_predicted = 0
                            else:
                                not_predicted_idx = torch.max(output_tmp, dim=1).values < cur_threshold
                                n_not_predicted = torch.sum(not_predicted_idx).item()
                            n_not_predicted_per_stage[i_subnet - 1] += n_not_predicted
                            '''
                            wanna know accuracies of all the subnets even if their predictions aren't used
                            => no breaking
                            '''
                            # if n_not_predicted == 0:
                            #     break

                            if resolutions_list[i_subnet] != resolutions_list[i_subnet - 1]:
                                if images_orig is None:
                                    images_orig = torch.clone(images)
                                r = resolutions_list[i_subnet]
                                images = torchvision.transforms.functional.resize(images_orig, (r, r))

                        if i_subnet == 0:
                            out_logits = subnet(images)
                            output_cur_softmaxed = torch.nn.functional.softmax(out_logits, dim=1)
                        else:
                            out_logits = subnet(images)
                            if len(out_logits.shape) < 2:  # a single image is left in the batch, need to fix dim # wait, because I want per-subnet accuracies I pass the whole batch through the net, so this isn't necessary?
                                out_logits = out_logits[None, ...]
                            output_cur_softmaxed = torch.nn.functional.softmax(out_logits, dim=1)

                        if i_subnet == 0:
                            output = output_cur_softmaxed
                        else:
                            if n_not_predicted > 0: # if 0, actual predictions are not modified
                                n_nets_used_in_cascade = i_subnet + 1
                                coeff1 = ((n_nets_used_in_cascade - 1) / n_nets_used_in_cascade)
                                coeff2 = (1 / n_nets_used_in_cascade)
                                output_tmp[not_predicted_idx] = coeff1 * output_tmp[not_predicted_idx] \
                                                                + coeff2 * output_cur_softmaxed[idx_more_predictions_needed][not_predicted_idx]
                                # need "output_tmp" because in pytorch "a[x][y] = z" doesn't modify "a".
                                output[idx_more_predictions_needed] = output_tmp

                        if if_ensemble_perf_per_member:
                            acc1 = accuracy(output_cur_softmaxed.detach(), labels, topk=(1,))
                            acc1 = get_scalar_from_accuracy(acc1)
                            # the line below caused a bug because I sorted the subnets by their desired position
                            # the fix is done at the very end because I want the numbering to be consistent,
                            # i.e. within the loop the subnets are sorted by their desired position.
                            metric_dict_val[f'top1_s{i_subnet}'].update(acc1, output.size(0))

                    loss = val_criterion(output, labels)
                    acc1 = accuracy(output, labels, topk=(1,))
                    acc1 = get_scalar_from_accuracy(acc1)
                    metric_dict_val['top1'].update(acc1, output.size(0))

                    losses_val.update(loss.item(), images.size(0))
                    n_input_channels = images.size(1)
                    tqdm_postfix = {'l': losses_val.avg,
                                    **{key: metric_dict_val[key].avg for key in metric_dict_val},
                                    'i': images.size(2)}
                    if thresholds is not None:
                        tqdm_postfix['not_pr'] = n_not_predicted_per_stage
                        tqdm_postfix['thr'] = thresholds
                    t.set_postfix(tqdm_postfix)
                    t.update(1)

        metric = metric_dict_val['top1'].avg
        top1_err = utils.get_metric_complement(metric)
        resolution_for_flops = resolutions_list
        info = get_net_info(subnets, (n_input_channels, resolution_for_flops, resolution_for_flops),
                            measure_latency=None, print_info=False, clean=True, lut=None, if_dont_sum=if_cascade)
        if not if_cascade:
            complexity = info[sec_obj]
        else:
            flops_per_stage = info[sec_obj]
            n_images_total = len(run_config.valid_loader.dataset)
            true_flops = flops_per_stage[0] + sum(
                [n_not_predicted / n_images_total * flops for (n_not_predicted, flops) in
                 zip(n_not_predicted_per_stage, flops_per_stage[1:])])
            complexity = true_flops
        del subnet

        to_return = top1_err, complexity

        if if_ensemble_perf_per_member:
            top1_err_per_member = []
            for i_subnet in range(len(subnets)):
                metric_cur = metric_dict_val[f'top1_s{i_subnet}'].avg

                top1_err_cur = utils.get_metric_complement(metric_cur)
                top1_err_per_member.append(top1_err_cur)
            # fixing the bug that arose because subnets were sorted by resolution but the code that gets
            # the output of this assumes sorting by supernet
            top1_err_per_member = np.array(top1_err_per_member)[reverse_idx].tolist()

            flops_per_member = np.array(flops_per_stage)[reverse_idx].tolist()

            to_return = (*to_return, (tuple(top1_err_per_member), tuple(flops_per_member)))
        else:
            to_return = (*to_return, None)

        return to_return

    @staticmethod
    def _extract_subnet_from_supernet(config_padded, get_engine, run_config, vld_batch_size, device):
        engine = get_engine(config_padded['w'])
        engine.set_active_subnet(ks=config_padded['ks'], e=config_padded['e'], d=config_padded['d'],
                                 w=config_padded['w'])
        resolution = config_padded['r']
        run_config.data_provider.collator_subtrain.set_resolutions([resolution])# for sub_train_loader
        run_config.data_provider.assign_active_img_size(resolution)  # if no training is done, active image size is not set
        st = time.time()
        data_loader_set_bn = run_config.random_sub_train_loader(2000, vld_batch_size, resolution)
        end = time.time()
        print(f'sub_train_loader time = {end-st}')
        subnet = engine.get_active_subnet(True)
        subnet.eval().to(device)
        # set BatchNorm for proper values for this subnet
        st = time.time()
        set_running_statistics(subnet, data_loader_set_bn)
        end = time.time()
        print(f'Setting BN time = {end-st}')
        return resolution, subnet

    @staticmethod
    def make_lambda_for_engine_creation(class_to_use, n_classes, use_gradient_checkpointing,
                                        dataset_name, search_space_name):
        def inner(supernet_path, run_config, to_cuda=True, device=None, if_create_optimizer=True):
            loaded_checkpoint = torch.load(supernet_path, map_location='cpu')
            n_in_channels = 3

            if search_space_name == 'ofa':
                if 'width_mult' in loaded_checkpoint:
                    width_mult = loaded_checkpoint['width_mult']
                else:
                    width_mult = 1.0 if 'w1.0' in supernet_path else 1.2 if 'w1.2' in supernet_path else None
                    assert width_mult is not None

                kernel_size = [3, 5, 7]
                exp_ratio = [3, 4, 6]
                depth = [2, 3, 4]
                engine = class_to_use(n_classes=n_classes, dropout_rate=0, width_mult=width_mult, ks_list=kernel_size,
                    expand_ratio_list=exp_ratio, depth_list=depth, if_use_gradient_checkpointing=use_gradient_checkpointing,
                                      n_image_channels=n_in_channels)

            elif search_space_name == 'alphanet':
                engine = class_to_use(n_classes=n_classes, if_use_gradient_checkpointing=use_gradient_checkpointing,
                                      n_image_channels=n_in_channels)

            elif search_space_name == 'proxyless':
                width_mult = 1.3

                kernel_size = [3, 5, 7]
                exp_ratio = [3, 4, 6]
                depth = [2, 3, 4]
                engine = class_to_use(n_classes=n_classes, dropout_rate=0, width_mult=width_mult, ks_list=kernel_size,
                    expand_ratio_list=exp_ratio, depth_list=depth, if_use_gradient_checkpointing=use_gradient_checkpointing,
                                      n_image_channels=n_in_channels)
            else:
                raise NotImplementedError

            if 'state_dict' in loaded_checkpoint:  # for the pretrained model
                init = loaded_checkpoint['state_dict']
            elif 'model_state_dict' in loaded_checkpoint:
                init = loaded_checkpoint['model_state_dict']
            else:
                raise ValueError

            if search_space_name == 'alphanet': #each key in the pretrained model starts with "module."
                init = {k.replace('module.', ''):v for k, v in init.items()}

            classifier_linear_name = 'classifier.linear'
            if classifier_linear_name + '.weight' not in init:
                classifier_linear_name += '.linear'

            loaded_classifier_weight_shape = init[classifier_linear_name + '.weight'].shape
            if (loaded_classifier_weight_shape[0] != n_classes):
                init[classifier_linear_name + '.weight'] = torch.rand((n_classes, loaded_classifier_weight_shape[1]))
                init[classifier_linear_name + '.bias'] = torch.rand((n_classes))

            engine.load_state_dict(init)

            if to_cuda:
                assert device is not None
                print(f'{device=}')
                engine.to(device)

            if if_create_optimizer:
                try:
                    net_params = engine.weight_parameters()
                except:
                    net_params = [param for param in engine.parameters() if param.requires_grad]

                optimizer = run_config.build_optimizer(net_params)
                if 'optimizer_state_dict' in loaded_checkpoint:
                    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
                print(optimizer)
            else:
                optimizer = None

            return engine, optimizer

        return inner
    
    def plot_archive(self, archive, c_top1_err, candidates, complexity, it, pred_for_candidates):
        plot = Scatter(legend=(True, {'loc': 'lower right'}), figsize=(12, 9))
        F = np.full((len(archive), 2), np.nan)
        F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
        F[:, 1] = get_metric_complement(np.array([x[1] for x in archive]))  # top-1 accuracy
        plot.add(F, s=15, facecolors='none', edgecolors='b', label='archive')
        F = np.full((len(candidates), 2), np.nan)
        proper_second_obj = np.array(complexity)
        F[:, 0] = proper_second_obj
        F[:, 1] = get_metric_complement(np.array(c_top1_err))
        plot.add(F, s=30, color='r', label='candidates evaluated')
        F = np.full((len(candidates), 2), np.nan)
        if not self.if_cascade:
            F[:, 0] = proper_second_obj
        else:
            F[:, 0] = pred_for_candidates[:, 1]
        F[:, 1] = get_metric_complement(pred_for_candidates[:, 0])
        plot.add(F, s=20, facecolors='none', edgecolors='g', label='candidates predicted')
        plot.plot_if_not_done_yet()
        plt.xlim(left=30)
        if self.dataset == 'cifar10':
            if np.median(F[:, 1]) > 85:
                plt.xlim(left=0, right=3000)
                plt.ylim(85, 100)
        elif self.dataset == 'cifar100':
            if np.median(F[:, 1]) > 70:
                plt.xlim(left=0, right=3000)
                plt.ylim(70, 90)
        elif self.dataset == 'imagenet':
            plt.xlim(left=0, right=2100)
            plt.ylim(64, 78)
        plot.save(os.path.join(self.path_logs, 'iter_{}.png'.format(it)))

def main(args):
    engine = NAT(args)
    engine.search()

    try:
        save_for_c_api_last_path = os.path.join(engine.path_logs, f'iter_{args["iterations"]}', 'save_for_c_api')
        os.remove(save_for_c_api_last_path)
    except:
        pass

    del engine
    gc.collect()
    torch.cuda.empty_cache()

default_kwargs = {
    'experiment_name': ['debug_run', 'location of dir to save'],
    'resume': [None, 'resume search from a checkpoint'],
    'sec_obj': ['flops', 'second objective to optimize simultaneously'],
    'iterations': [30, 'number of search iterations'],
    'n_doe': [100, 'number of architectures to sample initially '
                   '(I kept the old name which is a bit weird; "doe"=="design of experiment")'],
    'n_iter': [8, 'number of architectures to evaluate in each iteration'],
    'predictor': ['rbf', 'which accuracy predictor model to fit'],
    'data': ['/export/scratch3/aleksand/data/CIFAR/', 'location of the data corpus'],
    'dataset': ['cifar10', 'name of the dataset [imagenet, cifar10, cifar100, ...]'],
    'n_workers': [8, 'number of workers for dataloaders'],
    'vld_size': [10000, 'validation size'],
    'total_size': [None, 'train+validation size'],
    'trn_batch_size': [96, 'train batch size'],
    'vld_batch_size': [96, 'validation batch size '],
    'n_epochs': [5, 'test batch size for inference'],
    'supernet_path': [['/export/scratch3/aleksand/nsganetv2/data/ofa_mbv3_d234_e346_k357_w1.0'], 'list of paths to supernets'],
    'search_algo': ['nsga3', 'which search algo to use [NSGA-III, MO-GOMEA, random]'],
    'subset_selector': ['reference', 'which subset selector algo to use'],
    'init_with_nd_front_size': [0, 'initialize the search algorithm with subset of non-dominated front of this size'],
    'dont_check_duplicates': [False, 'if disable check for duplicates in search results'],
    'add_archive_to_candidates': [False, 'if a searcher should append archive to the candidates'],
    'sample_configs_to_train': [False, 'if instead of training selected candidates, a probability distribution '
                                       'should be constructed from archive, and sampled from (like in NAT)'],
    'random_seed': [42, 'random seed'],
    'n_warmup_epochs': [0, 'number of epochs for warmup'],
    'path_logs': ['/export/scratch3/aleksand/nsganetv2/logs/', 'Path to the logs folder'],
    'n_surrogate_evals': [800, 'Number of evaluations of the surrogate per meta-iteration'],
    'config_msunas_path': [None, 'Path to the yml file with all the parameters'],
    'gomea_exe': [None, 'Path to the mo-gomea executable file'],
    'alphabet': [['2'], 'Paths to text files (one per supernetwork) with alphabet size per variable'],
    'search_space': [['ensemble'], 'Supernetwork search space to use'],
    'store_checkpoint_freq': [1, 'Checkpoints will be stored for every x-th iteration'],
    'init_lr': [None, 'initial learning rate'],
    'ensemble_ss_names': [[], 'names of search spaces used in the ensemble'],
    'rbf_ensemble_size': [500, 'number of the predictors in the rbf_ensemble surrogate'],
    'cutout_size': [32, 'Cutout size. 0 == disabled'],
    'label_smoothing': [0.0, 'label smoothing coeff when doing classification'],
    'if_amp': [False, 'if train in mixed precision'],
    'use_gradient_checkpointing': [False, 'if use gradient checkpointing'],
    'lr_schedule_type': ['cosine', 'learning rate schedule; "cosine" is cyclic'],
    'if_cutmix': [False, 'if to use cutmix'],
    'weight_decay': [4e-5, ''],
    'if_center_crop': [True, 'if do center crop, or just resize to target size'],
    'auto_augment': ['rand-m9-mstd0.5', 'randaugment policy to use, or None to not use randaugment'],
    'resize_scale': [0.08, 'minimum resize scale in RandomResizedCrop, or None to not use RandomResizedCrop'],
    'search_goal': ['ensemble', 'Either "reproduce_nat" for reproducing NAT, or "ensemble" for everything else'],
}