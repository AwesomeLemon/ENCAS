import dill as pickle
import os

import numpy as np

from mo_gomea import MoGomeaCInterface
from utils import threshold_gene_to_value_moregranular as threshold_gene_to_value


def write_np_to_text_file_for_mo_gomea(path, arr):
    with open(path, 'wb') as f:
        np.savetxt(f, arr, delimiter=' ', newline='\n', header='', footer='', comments='# ', fmt='%d')
        # remove last empty line:
        NEWLINE_SIZE_IN_BYTES = -1
        f.seek(NEWLINE_SIZE_IN_BYTES, 2)
        f.truncate()


class MoGomeaWrapperEnsembleClassification:
    def __init__(self, alphabet, subnet_to_output_distrs, subnet_to_flops, labels, if_allow_noop, ensemble_size, **kwargs):
        super().__init__()

        self.n_evals = kwargs['n_evals']
        # gomea_exe_path = '/export/scratch3/aleksand/MO_GOMEA/exes/MO_GOMEA_default_ndinit_lb_lessoutput_intsolution_dontcountcache_usepythonpath'
        # gomea_exe_path = '/home/chebykin/MO_GOMEA/exes/MO_GOMEA_default_ndinit_lb_lessoutput_intsolution_dontcountcache_usepythonpath'
        gomea_exe_path = kwargs['gomea_exe_path']
        workdir_mo_gomea = kwargs['run_path']

        n_genes = ensemble_size
        if kwargs['search_goal'] == 'cascade':
            #dump alphabet
            alphabet = np.array([alphabet] * ensemble_size + [len(threshold_gene_to_value)] * (ensemble_size - 1))

            path_alphabet = os.path.join(workdir_mo_gomea, 'alphabet.txt')
            write_np_to_text_file_for_mo_gomea(path_alphabet, alphabet) # "+1" because mo-gomea wants non-inclusive upper bound
            n_genes = ensemble_size + (ensemble_size - 2) + 1 # net indices, threshold diff, baseline threshold

        # dump additional data
        path_data_dump = os.path.join(workdir_mo_gomea, 'data_dump_for_gomea')
        with open(path_data_dump, 'wb') as file_data_dump:
            pickle.dump({'if_allow_noop': if_allow_noop, 'subnet_to_flops': subnet_to_flops,
                         'labels_path': labels, 'output_distr_paths': subnet_to_output_distrs,
                         'search_goal': kwargs['search_goal']}, file_data_dump)

        self.mo_gomea = MoGomeaCInterface('EncasFitness', workdir_mo_gomea, path_data_dump, 2,
                                          n_genes=n_genes,
                                          alphabet=str(alphabet) if kwargs['search_goal'] != 'cascade' else path_alphabet,
                                          alphabet_lower_bound_path='0',
                                          gomea_executable_path=gomea_exe_path)

    def search(self, seed):
        genomes, objs = self.mo_gomea.search(self.n_evals, seed)

        # because in MO-Gomea I maximize "-flops" and "-error"
        objs *= -1
        return genomes, objs