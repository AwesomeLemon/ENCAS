import dill as pickle
import os

import numpy as np

from encas.encas_api import EncasAPI
from utils import threshold_gene_to_value_moregranular as threshold_gene_to_value, CsvLogger


class RandomSearchWrapperEnsembleClassification:
    def __init__(self, alphabet, subnet_to_output_distrs, subnet_to_flops, labels, if_allow_noop, ensemble_size, **kwargs):
        super().__init__()


        self.n_evals = kwargs['n_evals']
        workdir = kwargs['run_path']

        if kwargs['search_goal'] == 'cascade':
            self.alphabet = np.array([alphabet] * ensemble_size + [len(threshold_gene_to_value)] * (ensemble_size - 1))

        # dump additional data
        path_data_dump = os.path.join(workdir, 'data_dump_for_gomea')
        with open(path_data_dump, 'wb') as file_data_dump:
            pickle.dump({'if_allow_noop': if_allow_noop, 'subnet_to_flops': subnet_to_flops,
                         'labels_path': labels, 'output_distr_paths': subnet_to_output_distrs, 'search_goal': kwargs['search_goal']}, file_data_dump)

        self.fitness_api = EncasAPI(path_data_dump)
        self.logger = CsvLogger(workdir, 'random.csv')

    def search(self, seed, **kwargs):
        all_solutions = []
        all_objectives = []
        evals_performed = 0
        evaluated_solutions = set()
        while evals_performed < self.n_evals:
            solution = np.array([np.random.choice(omega_i, 1) for omega_i in self.alphabet])
            solution = solution[:, 0].tolist()
            while tuple(solution) in evaluated_solutions:
                solution = np.array([np.random.choice(omega_i, 1) for omega_i in self.alphabet])
                solution = solution[:, 0].tolist()

            top1_err, other_obj = self.fitness_api.fitness(solution)
            top1_err, other_obj = -top1_err, -other_obj  # because in the API I maximize "-obj"
            true_objs = (top1_err, other_obj)
            # [cur_solution_idx, elapsed_time, x, fitness_str]
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
        print(all_solutions)
        print(all_objectives)
        return all_solutions, all_objectives