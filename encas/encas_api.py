import glob
import pickle

import gzip
import numpy as np
import os
import torch

from utils import threshold_gene_to_value_moregranular as threshold_gene_to_value


class EncasAPI:
    def __init__(self, filename):
        self.use_cache = True

        kwargs = pickle.load(open(filename, 'rb'))
        self.if_allow_noop = kwargs['if_allow_noop']
        self.subnet_to_flops = kwargs['subnet_to_flops']
        self.search_goal = kwargs['search_goal'] # ensemble or cascade

        labels_path = kwargs['labels_path']
        self.labels = torch.tensor(np.load(labels_path)).cuda()

        output_distr_paths = kwargs['output_distr_paths']
        outputs_all = []
        for p in output_distr_paths:
            output_distr = []
            n_files = len(glob.glob(os.path.join(p, "*.npy.gz")))
            for j in range(n_files):
                print(os.path.join(p, f'{j}.npy.gz'))
                with gzip.GzipFile(os.path.join(p, f'{j}.npy.gz'), 'r') as f:
                    output_distr.append(np.asarray(np.load(f), dtype=np.float16)[None, ...])
            outputs_all += output_distr

        if self.if_allow_noop:
            output_distr_onenet_noop = np.zeros_like(output_distr[0])  # copy arbitrary one to get the shape
            if len(output_distr_onenet_noop.shape) < 3:
                output_distr_onenet_noop = output_distr_onenet_noop[None, ...]
            outputs_all = [output_distr_onenet_noop] + outputs_all

        if True: #pre-allocation & concatenation on CPU is helpful when there's not enough VRAM
            preallocated_array = np.zeros((len(outputs_all), outputs_all[-1].shape[1], outputs_all[-1].shape[2]), dtype=np.float16)
            np.concatenate((outputs_all), axis=0, out=preallocated_array)
            self.subnet_to_output_distrs = torch.tensor(preallocated_array).cuda()
        else:
            outputs_all = [torch.tensor(o).cuda() for o in outputs_all]
            self.subnet_to_output_distrs = torch.cat(outputs_all, dim=0)
        print(f'{self.subnet_to_output_distrs.shape=}')
        print(f'{len(self.subnet_to_flops)=}')

    def _fitness_ensemble(self, solution):
        solution_size = len(solution)

        nets_used = solution_size
        if self.if_allow_noop:
            n_noops = sum([1 if g == 0 else 0 for g in solution])
            if n_noops == nets_used:
                return (-100, -1e5)

            nets_used -= n_noops

        preds = torch.clone(self.subnet_to_output_distrs[solution[0]])

        for j in range(1, solution_size):
            preds_cur = self.subnet_to_output_distrs[solution[j]]
            preds += preds_cur

        preds /= nets_used
        output = torch.argmax(preds, 1)

        err = (torch.sum(self.labels != output).item() / len(self.labels)) * 100
        flops = sum([self.subnet_to_flops[solution[j]] for j in range(solution_size)])
        obj0_proper_form, obj1_proper_form = -err, -flops
        return (obj0_proper_form, obj1_proper_form)

    def _fitness_cascade(self, solution):
        max_n_nets = (len(solution) + 1) // 2
        solution_nets, solution_thresholds = solution[:max_n_nets], solution[max_n_nets:]

        n_nets_not_noops = max_n_nets
        if self.if_allow_noop:
            n_noops = sum([1 if g == 0 else 0 for g in solution_nets])
            if n_noops == n_nets_not_noops:
                return (-100, -1e5)

            n_nets_not_noops -= n_noops
        n_nets_used_in_cascade = 0

        preds = torch.tensor(self.subnet_to_output_distrs[solution_nets[0]])
        flops = self.subnet_to_flops[solution_nets[0]]
        n_nets_used_in_cascade += int(solution_nets[0] != 0)

        idx_more_predictions_needed = torch.ones(preds.shape[0], dtype=torch.bool)

        for j in range(1, max_n_nets):
            if solution_nets[j] == 0: # noop
                continue

            cur_threshold = threshold_gene_to_value[solution_thresholds[j - 1]]

            idx_more_predictions_needed[torch.max(preds, dim=1).values > cur_threshold] = False
            preds_tmp = preds[idx_more_predictions_needed] #preds_tmp is needed because I wanna do (in the end) x[idx1][idx2] = smth, and that doesn't modify the original x
            not_predicted_idx = torch.max(preds_tmp, dim=1).values <= cur_threshold
            # not_predicted_idx = torch.max(preds, dim=1).values < cur_threshold
            n_not_predicted = torch.sum(not_predicted_idx).item()
            # print(f'{n_not_predicted=}')
            if n_not_predicted == 0:
                break

            n_nets_used_in_cascade += 1 #it's guaranteed to not be a noop
            preds_cur = self.subnet_to_output_distrs[solution_nets[j]]
            if_average_outputs = True
            if if_average_outputs:
                coeff1 = (n_nets_used_in_cascade - 1) / n_nets_used_in_cascade # for the current predictions that may already be an average
                coeff2 = 1 / n_nets_used_in_cascade # for the predictions of the new model

                preds_tmp[not_predicted_idx] = coeff1 * preds_tmp[not_predicted_idx] \
                                           + coeff2 * preds_cur[idx_more_predictions_needed][not_predicted_idx]
                preds[idx_more_predictions_needed] = preds_tmp
            else:
                preds_tmp[not_predicted_idx] = preds_cur[idx_more_predictions_needed][not_predicted_idx]
                preds[idx_more_predictions_needed] = preds_tmp

            flops += self.subnet_to_flops[solution_nets[j]] * (n_not_predicted / len(self.labels))

        output = torch.argmax(preds, dim=1)

        err = (torch.sum(self.labels != output).item() / len(self.labels)) * 100

        obj0_proper_form, obj1_proper_form = -err, -flops
        return (obj0_proper_form, obj1_proper_form)

    def fitness(self, solution):
        # st = time.time()
        solution = [int(x) for x in solution]
        if self.search_goal == 'ensemble':
            res = self._fitness_ensemble(solution)
        elif self.search_goal == 'cascade':
            res = self._fitness_cascade(solution)
        else:
            raise NotImplementedError(f'Unknown {self.search_goal=}')
        # ed = time.time()
        # self.avg_time.update(ed - st)
        # if self.avg_time.count % 500 == 0:
        #     print(f'Avg fitness time is {self.avg_time.avg} @ {self.avg_time.count}')

        return res