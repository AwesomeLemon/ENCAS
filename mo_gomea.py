import os
import pandas as pd
import numpy as np

from utils import capture_subprocess_output
from pathlib import Path

class MoGomeaCInterface():
    name = 'mo_gomea'
    def __init__(self, api_name, path, path_data_for_c_api, n_objectives=2, n_genes=10, alphabet='2',
                 alphabet_lower_bound_path='0', init_path=None,
                 gomea_executable_path='/export/scratch3/aleksand/MO_GOMEA/cmake-build-debug-remote/MO_GOMEA'):
        super().__init__()
        self.api_name = api_name
        self.path = path
        self.path_data_for_c_api = path_data_for_c_api
        Path(self.path).mkdir(exist_ok=True) # need to create it before calling the C executable
        # self.logger = CsvLogger(self.path, self.name + '.csv')
        self.n_objectives = n_objectives
        self.n_elitists = 10000#40
        self.n_genes = n_genes
        self.alphabet = alphabet
        self.alphabet_lower_bound_path = alphabet_lower_bound_path
        self.init_path = init_path
        self.gomea_executable_path = gomea_executable_path


    def search(self, n_evaluations, seed):
        n_inbetween_log_files = 10
        log_interval = n_evaluations // n_inbetween_log_files
        subprocess_params = [str(self.gomea_executable_path), '-p', '5', str(self.n_objectives),
                     str(self.n_genes), str(self.n_elitists), str(n_evaluations),
                     str(log_interval), str(self.path), str(self.api_name), str(seed),
                     str(self.path_data_for_c_api), self.alphabet, self.alphabet_lower_bound_path]
        if self.init_path is not None:
            subprocess_params.append(self.init_path)
        print(' '.join(subprocess_params))
        output = capture_subprocess_output(subprocess_params)
        df = pd.read_csv(os.path.join(self.path, 'elitist_archive_generation_final.dat'), sep=' ', header=None)
        genomes = np.array([x.split(',')[:-1] for x in df.iloc[:, -1]], dtype=np.int)
        obj0 = np.array(df.iloc[:, 0])
        obj1 = np.array(df.iloc[:, 1])
        if self.n_objectives > 2:
            obj2 = np.array(df.iloc[:, 2])
            objs_final = np.vstack([obj0, obj1, obj2]).T
            return genomes, objs_final  # shape is (n_individuals, n_objs)
        return genomes, np.vstack([obj0, obj1]).T