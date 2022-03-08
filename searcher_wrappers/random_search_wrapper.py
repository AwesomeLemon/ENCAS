import os
from pathlib import Path

from networks.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from networks.ofa_mbv3_my import OFAMobileNetV3My
from networks.proxyless_my import OFAProxylessNASNetsMy
from searcher_wrappers.base_wrapper import BaseSearcherWrapper
import numpy as np

from utils import CsvLogger, get_net_info, SupernetworkWrapper


class RandomSearchWrapper(BaseSearcherWrapper):
    def __init__(self, search_space, sec_obj, path_logs,
                 n_classes, supernet_paths, n_evals, if_add_archive_to_candidates,
                 **kwargs):
        super().__init__()
        self.search_space = search_space
        self.n_obj = 2
        self.problem = RandomSearch(
            self.search_space, None, n_classes, sec_obj,
            supernet_paths, self.n_obj, n_evals, **kwargs)
        self.if_add_archive_to_candidates = if_add_archive_to_candidates
        self.path_logs = path_logs


    def search(self, archive, predictor, iter_current, **kwargs):
        workdir = os.path.join(self.path_logs, f'iter_{iter_current}')
        Path(workdir).mkdir(exist_ok=True)

        # initialize the candidate finding optimization problem
        self.problem.predictor = predictor
        self.problem.logger = CsvLogger(workdir, 'random.csv')

        genomes, objs = self.problem.perform_random_search()

        if self.if_add_archive_to_candidates:
            archive_unzipped = list(zip(*archive))
            configs_archive = archive_unzipped[0]
            if 'rbf_ensemble_per_ensemble_member' not in predictor.name:
                objs_archive = np.vstack(archive_unzipped[1:]).T  # archive needs to contain all objective functions
            else:
                objs_archive = np.vstack(archive_unzipped[1:-1]).T  # ignore objective function that is the tuples of ensemblee metrics
                # we can do it because this objs_archive will be used exclusively for subset selection, and we probably don't wanna select on that
            genomes_archive = [self.search_space.encode(c) for c in configs_archive]
            # need to check for duplicates
            for i in range(genomes.shape[0]):
                # if genomes[i] not in genomes_archive:
                if not any((genomes[i] == x).all() for x in genomes_archive):
                    genomes_archive.append(genomes[i])
                    objs_archive = np.vstack((objs_archive, objs[i]))

            genomes = np.array(genomes_archive)
            objs = np.array(objs_archive)

            np.save(os.path.join(workdir, 'genomes.npy'), genomes)
            np.save(os.path.join(workdir, 'objs.npy'), objs)

        return genomes, objs

class RandomSearch:

    def __init__(self, search_space, predictor, n_classes, sec_obj='flops', supernet_paths=None,
                 n_obj=2, n_evals=1, alphabet=None, **kwargs):
        # super().__init__(n_var=46, n_obj=n_obj, n_constr=0, type_var=np.int)
        # super().__init__(n_var=36, n_obj=2, n_constr=0, type_var=np.int)# ACHTUNG! modified for the original binary

        self.ss = search_space
        self.predictor = predictor
        print('Achtung! lower bound')
        xl = np.array(kwargs['alphabet_lb'])#np.array([0, 0] + [1, 1, 0, 0] * 5)#np.zeros(self.n_var)
        xu = np.array(alphabet) #- 1 # remove "-1" because np.random.choise has exclusive upper bound
        self.alphabet_lb_and_ub = list(zip(xl, xu))
        self.sec_obj = sec_obj
        self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        search_space_name = kwargs['search_space_name']
        self.search_space_name = search_space_name
        dataset = kwargs['dataset']
        self.n_image_channels = kwargs['n_image_channels']
        assert self.search_space_name == 'ensemble'

        # assume supernet_paths is a list of paths, 1 per supernet
        ensemble_ss_names = kwargs['ensemble_ss_names']
        ss_name_to_class = {'alphanet': AttentiveNasDynamicModel, 'ofa': OFAMobileNetV3My,
                                'proxyless': OFAProxylessNASNetsMy}

        classes_to_use = [ss_name_to_class[ss_name] for ss_name in ensemble_ss_names]
        self.evaluators = [SupernetworkWrapper(n_classes=n_classes, model_path=supernet_path,
                                               engine_class_to_use=encoder_class,
                                               n_image_channels=self.n_image_channels, if_ignore_decoder=False, dataset=dataset,
                                               search_space_name=ss_name)
                           for supernet_path, ss_name, encoder_class in zip(supernet_paths, ensemble_ss_names, classes_to_use)]
        self.logger = None
        self.n_evals = n_evals


    def _evaluate(self, solution):
        if self.sec_obj in ['flops', 'cpu', 'gpu']:

            config = self.ss.decode(solution)
            sec_objs = []
            for conf, evaluator in zip(config, self.evaluators):
                subnet, _ = evaluator.sample({'ks': conf['ks'], 'e': conf['e'], 'd': conf['d'], 'w': conf['w']})
                info = get_net_info(subnet, (self.n_image_channels, conf['r'], conf['r']),
                                    measure_latency=self.sec_obj, print_info=False, clean=True, lut=self.lut)
                sec_objs.append(info[self.sec_obj])

            input_acc = np.array(solution)[np.newaxis, :]
            solution_reencoded_sep = self.ss.encode(config, if_return_separate=True)
            input_flops = np.concatenate(
                [sol_sep[-2:] for sol_sep in solution_reencoded_sep] + [[int(f) for f in sec_objs]])[np.newaxis, :]

            top1_err_and_other_obj = self.predictor.predict({'for_acc': input_acc, 'for_flops': input_flops})[0]
            top1_err = top1_err_and_other_obj[0]
            other_obj = top1_err_and_other_obj[1]

        return top1_err, other_obj

    def perform_random_search(self):
        all_solutions = []
        all_objectives = []
        evals_performed = 0
        evaluated_solutions = set()
        while evals_performed < self.n_evals:
            solution = np.array([np.random.choice(range(lb, ub), 1) for lb, ub in self.alphabet_lb_and_ub])
            solution = solution[:, 0].tolist()
            while tuple(solution) in evaluated_solutions:
                solution = np.array([np.random.choice(range(lb, ub), 1) for lb, ub in self.alphabet_lb_and_ub])
                solution = solution[:, 0].tolist()

            top1_err, other_obj = self._evaluate(solution)
            true_objs = (top1_err, other_obj)
            true_objs_str = str(true_objs).replace(' ', '')
            self.logger.log([evals_performed, 0, ','.join([str(s) for s in solution]), true_objs_str])
            evals_performed += 1
            # print(f'{evals_performed}: New solution! {true_objs=}')
            if evals_performed % 1000 == 0:
                print(f'{evals_performed=}')
            all_solutions.append(solution)
            all_objectives.append(list(true_objs))
            evaluated_solutions.add(tuple(solution))

        all_solutions = np.vstack(all_solutions)
        all_objectives = np.array(all_objectives)
        return all_solutions, all_objectives