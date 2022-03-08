import numpy as np
import time

from utils import set_seed
from utils import CsvLogger
from nat_api import NatAPI
from encas.encas_api import EncasAPI


def alphabet_to_list(alphabet, n_variables):
    if alphabet.isnumeric():
        return [int(alphabet) for _ in range(n_variables)]
    file = open(alphabet, 'r')
    alphabetSizes = file.readline().split(' ')
    file.close()

    return [int(alphabetSizes[i]) for i in range(n_variables)]


class Logger:
    def __init__(self, folder):
        self.folder = folder
        self.solutions_cache = {}
        # self.solutionsCounter = {}
        self.start_time = time.time()

        # self.file = open('%s/optimization.txt' % self.folder, 'w', buffering=1)
        # self.file.write('#Evals time solution fitness\n')
        # file.close()

        self.csv_logger = CsvLogger(self.folder, 'gomea.csv')
        self.eval_cnt = 0

    def elapsed_time(self):
        return time.time() - self.start_time

    def return_solution(self, x):
        return self.solutions_cache.get(x, None)

    def solution_to_str(self, arr):
        x = [str(i) for i in arr]
        x = ''.join(x)
        return x

    def solution_to_str_commas(self, arr):
        x = [str(i) for i in arr]
        x = ','.join(x)
        return x

    def write(self, x, fitness):
        if x not in self.solutions_cache:
            self.solutions_cache[x] = fitness

        elapsed_time = time.time() - self.start_time
        cur_solution_idx = self.eval_cnt
        self.eval_cnt += 1

        fitness_str = str(fitness).replace(' ', '')
        self.csv_logger.log([cur_solution_idx, elapsed_time, x, fitness_str])



class FitnessFunction():
    def __init__(self, folder, filename, n_variables, alphabet, random_seed):
        self.logger = Logger(folder)
        self.numberOfVariables = int(n_variables)
        self.alphabet = alphabet_to_list(alphabet, n_variables)
        self.filename = filename

    def fitness(self, x):
        pass

class FitnessFunctionAPIWrapper(FitnessFunction):

    def __init__(self, folder, filename, n_variables, alphabet, random_seed, if_count_zeros=True):
        super().__init__(folder, filename, n_variables, alphabet, random_seed)
        self.api = None # descendants will need to initialize the API
        self.if_count_zeros = if_count_zeros
        set_seed(random_seed)

    def fitness(self, solution):
        assert isinstance(solution, list) or isinstance(solution, tuple) or isinstance(solution, np.ndarray)
        solution = np.array(solution).astype(np.int32)
        solution_str = self.logger.solution_to_str(solution)

        if self.api.use_cache:
            find = self.logger.return_solution(solution_str)
            if find != None:
                return find
        score = self.api.fitness(solution)

        if self.if_count_zeros or score != 0:
            self.logger.write(solution_str, score)

        return score

class FitnessFunctionAPIWrapperWithTransparentCaching(FitnessFunction):
    '''
    difference to FitnessFunctionAPIWrapper: the first value in the returned tuple is True if cache was used
    (i.e. no new evaluation). It is cast to long because that's easier to handle on the C side
    '''

    def __init__(self, folder, filename, n_variables, alphabet, random_seed, if_count_zeros=True):
        super().__init__(folder, filename, n_variables, alphabet, random_seed)
        self.api = None # descendants will need to initialize the API
        self.if_count_zeros = if_count_zeros
        set_seed(random_seed)

    def fitness(self, solution):
        assert isinstance(solution, list) or isinstance(solution, tuple) or isinstance(solution, np.ndarray)
        solution = np.array(solution).astype(np.int32)
        solution_str = self.logger.solution_to_str_commas(solution)

        if self.api.use_cache:
            find = self.logger.return_solution(solution_str)
            if find != None:
                return (int(True),) + find
        score = self.api.fitness(solution)

        if self.if_count_zeros or score != 0:
            self.logger.write(solution_str, score)

        return (int(False),) + score

class NatFitness(FitnessFunctionAPIWrapperWithTransparentCaching):
    def __init__(self, folder, filename, n_variables, alphabet, random_seed):
        super().__init__(folder, filename, n_variables, alphabet, random_seed)
        self.api = NatAPI(filename)

class EncasFitness(FitnessFunctionAPIWrapperWithTransparentCaching):
    def __init__(self, folder, filename, n_variables, alphabet, random_seed):
        super().__init__(folder, filename, n_variables, alphabet, random_seed)
        self.api = EncasAPI(filename)