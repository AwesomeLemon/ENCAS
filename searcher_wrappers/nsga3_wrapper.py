import os
import time
from pathlib import Path

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from networks.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from networks.ofa_mbv3_my import OFAMobileNetV3My
from networks.proxyless_my import OFAProxylessNASNetsMy
from utils import get_net_info, SupernetworkWrapper
from searcher_wrappers.base_wrapper import BaseSearcherWrapper
from pymoo.model.problem import Problem
import numpy as np
from pymoo.factory import get_crossover, get_mutation, get_reference_directions
from pymoo.optimize import minimize


from pymoo.algorithms.nsga3 import NSGA3

class Nsga3Wrapper(BaseSearcherWrapper):
    def __init__(self, search_space, sec_obj, path_logs,
                 n_classes, supernet_paths, n_evals, if_add_archive_to_candidates,
                 **kwargs):
        super().__init__()
        self.search_space = search_space
        self.n_obj = 2
        self.problem = NatProblem(self.search_space, None, n_classes, sec_obj, supernet_paths,
                                  self.n_obj, **kwargs)
        self.pop_size = 100
        self.n_gen = n_evals // self.pop_size
        self.if_add_archive_to_candidates = if_add_archive_to_candidates
        self.path_logs = path_logs
        self.ref_dirs = get_reference_directions("riesz", self.n_obj, 100)
        print(f'{self.ref_dirs.shape=}')


    def search(self, archive, predictor, iter_current, **kwargs):
        workdir = os.path.join(self.path_logs, f'iter_{iter_current}')
        Path(workdir).mkdir(exist_ok=True)
        F = np.column_stack(list(zip(*archive))[1:])
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        archive_encoded = np.array([self.search_space.encode(x[0]) for x in archive])
        nd_X = archive_encoded[front]

        # initialize the candidate finding optimization problem
        self.problem.predictor = predictor

        method = NSGA3(
            pop_size=self.pop_size, ref_dirs=self.ref_dirs, sampling=archive_encoded,
            crossover=get_crossover("int_ux", prob=0.9),
            mutation=get_mutation("int_pm", prob=0.1, eta=1.0),
            eliminate_duplicates=True
        )

        res = minimize(self.problem, method, termination=('n_gen', self.n_gen), save_history=False, verbose=True)
        genomes = res.pop.get('X')
        objs = res.pop.get('F')

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
                if not any((genomes[i] == x).all() for x in genomes_archive):
                    genomes_archive.append(genomes[i])
                    objs_archive = np.vstack((objs_archive, objs[i]))

            genomes = np.array(genomes_archive)
            objs = np.array(objs_archive)

            # useful for debugging:
            np.save(os.path.join(workdir, 'genomes.npy'), genomes)
            np.save(os.path.join(workdir, 'objs.npy'), objs)

        return genomes, objs

class NatProblem(Problem):
    '''
    an optimization problem for pymoo
    '''
    def __init__(self, search_space, predictor, n_classes, sec_obj='flops', supernet_paths=None,
                 n_obj=2, alphabet=None, **kwargs):
        super().__init__(n_var=len(alphabet), n_obj=n_obj, n_constr=0, type_var=np.int)
        self.ss = search_space
        self.predictor = predictor
        self.xl = np.array(kwargs['alphabet_lb'])
        self.xu = np.array(alphabet, dtype=np.float) - 1 # "-1" because pymoo wants inclusive range
        self.sec_obj = sec_obj
        self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        search_space_name = kwargs['search_space_name']
        self.search_space_name = search_space_name
        dataset = kwargs['dataset']
        self.n_image_channels = kwargs['n_image_channels']
        if search_space_name == 'reproduce_nat':
            ev1_0 = SupernetworkWrapper(n_classes=n_classes, model_path=supernet_paths[0],
                                        engine_class_to_use=OFAMobileNetV3My,
                                        n_image_channels=self.n_image_channels, if_ignore_decoder=True, dataset=dataset,
                                        search_space_name='ofa')
            ev1_2 = SupernetworkWrapper(n_classes=n_classes, model_path=supernet_paths[1],
                                        engine_class_to_use=OFAMobileNetV3My,
                                        n_image_channels=self.n_image_channels, if_ignore_decoder=True, dataset=dataset,
                                        search_space_name='ofa')
            self.get_engine = lambda config: {
                1.0: ev1_0,
                1.2: ev1_2,
            }[config['w']]
        elif search_space_name == 'ensemble':
            # assume supernet_paths is a list of paths, 1 per supernet
            ensemble_ss_names = kwargs['ensemble_ss_names']
            # since I don't use NSGA-3 for joint training & search, there should be only one supernet
            assert len(ensemble_ss_names) == 1
            ss_name_to_class = {'alphanet': AttentiveNasDynamicModel, 'ofa': OFAMobileNetV3My,
                                'proxyless': OFAProxylessNASNetsMy}

            classes_to_use = [ss_name_to_class[ss_name] for ss_name in ensemble_ss_names]
            self.evaluators = [SupernetworkWrapper(n_classes=n_classes, model_path=supernet_path,
                                                   engine_class_to_use=encoder_class,
                                                   n_image_channels=self.n_image_channels, if_ignore_decoder=False,
                                                   dataset=dataset, search_space_name=ss_name)
                                   for supernet_path, ss_name, encoder_class in
                                   zip(supernet_paths, ensemble_ss_names, classes_to_use)]

    def _evaluate(self, x, out, *args, **kwargs):
        st = time.time()
        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.predictor.predict(x)[:, 0]
        for i, (_x, err) in enumerate(zip(x, top1_err)):
            config = self.ss.decode(_x)

            if self.search_space_name == 'reproduce_nat':
                subnet, _ = self.get_engine(config).sample({'ks': config['ks'], 'e': config['e'],
                                                            'd': config['d'], 'w':config['w']})
                info = get_net_info(subnet, (self.n_image_channels, config['r'], config['r']),
                                    measure_latency=self.sec_obj, print_info=False, clean=True, lut=self.lut)
                f[i, 1] = info[self.sec_obj]
            else:
                sec_obj_sum = 0
                for conf, evaluator in zip(config, self.evaluators):
                    subnet, _ = evaluator.sample({'ks': conf['ks'], 'e': conf['e'], 'd': conf['d'], 'w':conf['w']})
                    info = get_net_info(subnet, (self.n_image_channels, conf['r'], conf['r']),
                                        measure_latency=self.sec_obj, print_info=False, clean=True, lut=self.lut)
                    sec_obj_sum += info[self.sec_obj]
                f[i, 1] = sec_obj_sum

            f[i, 0] = err

        out["F"] = f
        ed = time.time()
        print(f'Fitness time = {ed - st}')