import numpy as np
from pymoo.algorithms.nsga3 import ReferenceDirectionSurvival, get_extreme_points_c, get_nadir_point, \
    associate_to_niches, calc_niche_count, niching
from pymoo.factory import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from subset_selectors.base_subset_selector import BaseSubsetSelector
from matplotlib import pyplot as plt

class ReferenceBasedSubsetSelector(BaseSubsetSelector):
    '''
    same as in nsga3
    '''
    def __init__(self, n_select, **kwargs):
        super().__init__()
        self.n_select = n_select
        ref_dirs = get_reference_directions("riesz", kwargs['n_objs'], 100)
        self.selector = ReferenceDirectionSurvivalMy(ref_dirs)


    def select(self, archive, objs):
        # objs_cur shape is (n_archs, n_objs)
        objs = np.copy(objs)

        n_total = objs.shape[0]
        if n_total > self.n_select:
            indices_selected = self.selector._do(None, objs, self.n_select)
            print(f'rbf_ensemble: Selected {np.sum(indices_selected)} indices properly')
        else:
            indices_selected = [True] * n_total
            print(f'rbf_ensemble: Selected {n_total} indices by default')

        return indices_selected


class ReferenceDirectionSurvivalMy(ReferenceDirectionSurvival):
    '''
    Modified to work with np archives instead of the "population" structures.
    Behaviourally is the same as ReferenceDirectionSurvival.
    '''
    def __init__(self, ref_dirs):
        super().__init__(ref_dirs)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, problem, objs, n_survive, D=None, **kwargs):
        n_total = objs.shape[0]
        indices_selected = np.array([False] * n_total)

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, objs)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, objs)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(objs, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(objs[non_dominated, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(objs, axis=0)
        worst_of_front = np.max(objs[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        indices_selected[I] = True

        # update the front indices for the current population
        new_idx_to_old_idx = {}
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                new_idx_to_old_idx[counter] = fronts[i][j]
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(objs[indices_selected], self.ref_dirs, self.ideal_point, self.nadir_point)

        # if we need to select individuals to survive
        if len(objs[indices_selected]) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(objs[indices_selected][last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))

            # only survivors need to remain active
            indices_selected[:] = False
            indices_selected[[new_idx_to_old_idx[s] for s in survivors]] = True

        return indices_selected

if __name__ == '__main__':
    gss = ReferenceBasedSubsetSelector(5, n_objs=2)
    objs_cur = np.array([[0, 0], [0.2, 1.5], [0.9, 0],
                         [1, 2], [2, 4], [3, 3],
                         [-0.5, -0.5], [0.5, 0.7], [0.7, 0.5]])
    plt.scatter(objs_cur[:, 0], objs_cur[:, 1])
    idx = gss.select(None, objs_cur)
    print(objs_cur[idx])
    plt.scatter(objs_cur[idx, 0], objs_cur[idx, 1])
    plt.show()