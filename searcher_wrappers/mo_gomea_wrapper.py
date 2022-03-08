import os
import pickle
from pathlib import Path

import numpy as np

from searcher_wrappers.base_wrapper import BaseSearcherWrapper
from mo_gomea import MoGomeaCInterface
from utils import get_metric_complement
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class MoGomeaWrapper(BaseSearcherWrapper):
    def __init__(self, search_space, sec_obj, path_logs,
                 n_classes, supernet_paths, n_evals, if_add_archive_to_candidates,
                 **kwargs):
        super().__init__()
        self.n_objectives = 2
        self.path_logs = path_logs
        self.n_classes = n_classes
        self.supernet_paths = supernet_paths
        self.n_evals = n_evals
        self.sec_obj = sec_obj
        self.if_add_archive_to_candidates = if_add_archive_to_candidates
        self.search_space = search_space

        self.alphabet_name = kwargs['alphabet_name'] # actually, many names
        self.alphabet_size = len(kwargs['alphabet'])

        # construct a dynamic alphabet for the cascade
        self.alphabet = kwargs['alphabet']
        self.alphabet_lb = kwargs['alphabet_lb']
        self.alphabet_path = os.path.join(self.path_logs, 'dynamic_ensemble_alphabet.txt')
        with open(self.alphabet_path, mode='w', newline='') as f:
            to_print = ''
            for x in self.alphabet:
                to_print += f'{x} '
            to_print = to_print.rstrip()
            f.write(to_print)
            f.flush()

        self.alphabet_lower_bound_path = os.path.join(self.path_logs, 'dynamic_ensemble_alphabet_lb.txt')
        with open(self.alphabet_lower_bound_path, mode='w', newline='') as f:
            to_print = ''
            for x in self.alphabet_lb:
                to_print += f'{x} '
            to_print = to_print.rstrip()
            f.write(to_print)
            f.flush()

        if 'init_with_nd_front_size' in kwargs:
            self.init_with_nd_front_size = kwargs['init_with_nd_front_size']
        else:
            self.init_with_nd_front_size = 0
        if 'gomea_exe_path' not in kwargs:
            raise ValueError('Need to pass a path to the appropriate MO-GOMEA executable')
        self.gomea_exe_path = kwargs['gomea_exe_path']
        self.n_image_channels = kwargs['n_image_channels']
        self.dataset = kwargs['dataset']
        self.search_space_name = kwargs['search_space_name']
        self.ensemble_ss_names = kwargs['ensemble_ss_names']
        assert self.gomea_exe_path is not None


    def search(self, archive, predictor, iter_current, **kwargs):
        workdir_mo_gomea = os.path.join(self.path_logs, f'iter_{iter_current}')
        Path(workdir_mo_gomea).mkdir(exist_ok=True)

        archive_path = os.path.join(self.path_logs, f'iter_{iter_current - 1}.stats')

        path_data_for_c_api = os.path.join(workdir_mo_gomea, 'save_for_c_api')
        with open(path_data_for_c_api, 'wb') as file_data_for_c_api:
            pickle.dump((predictor, self.n_classes, self.supernet_paths, archive_path, self.sec_obj, '',
                         self.alphabet_name, self.n_image_channels, self.dataset, self.search_space_name,
                         self.ensemble_ss_names, ''),
                    file_data_for_c_api)

        path_init = None
        if self.init_with_nd_front_size > 0:
            F = np.column_stack(list(zip(*archive))[1:])
            front = NonDominatedSorting().do(F, only_non_dominated_front=True)
            nd_X = np.array([self.search_space.encode(x[0]) for x in archive], dtype=np.int)[front]
            n_to_use_for_init = self.init_with_nd_front_size#5#16
            if nd_X.shape[0] > n_to_use_for_init:
                chosen = np.random.choice(nd_X.shape[0], n_to_use_for_init, replace=False)
            else:
                chosen = np.ones((nd_X.shape[0]), dtype=bool)
            idx = np.zeros((nd_X.shape[0]), dtype=np.bool)
            idx[chosen] = True
            path_init = os.path.join(workdir_mo_gomea, 'init_nd_front')
            with open(path_init, 'wb') as f:
                np.savetxt(f, nd_X[idx], delimiter=' ', newline='\n', header='', footer='', comments='# ', fmt='%d')
                # remove last empty line:
                NEWLINE_SIZE_IN_BYTES = -1
                f.seek(NEWLINE_SIZE_IN_BYTES, 2)
                f.truncate()

        mo_gomea = MoGomeaCInterface('NatFitness', workdir_mo_gomea, path_data_for_c_api, self.n_objectives,
                                     n_genes=self.alphabet_size,
                                     alphabet=self.alphabet_path,
                                     alphabet_lower_bound_path=self.alphabet_lower_bound_path,
                                     init_path=path_init, gomea_executable_path=self.gomea_exe_path)
        genomes, objs = mo_gomea.search(n_evaluations=self.n_evals, seed=kwargs['seed'])

        if self.sec_obj in ['flops', 'cpu', 'gpu']:
            objs[:, 1] *= -1  # because in MO-Gomea I maximize "-flops"

        objs[:, 0] = get_metric_complement(objs[:, 0]) # because in MO-Gomea I maximize objective, not minimize error

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
            objs = objs_archive

        # delete 'save_for_c_api' file of the PREVIOUS iteration (otherwise problems when process is interrupted
        # after save_for_c_api has been deleted but before a new one is created)
        if iter_current > 1:
            data_prev = os.path.join(self.path_logs, f'iter_{iter_current - 1}', 'save_for_c_api')
            try:
                os.remove(data_prev)
            except:
                pass

        return genomes, objs