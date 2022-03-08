# implementation of the algorithm from the paper http://proceedings.mlr.press/v80/streeter18a/streeter18a.pdf
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import utils


class GreedySearchWrapperEnsembleClassification:
    def __init__(self, alphabet, subnet_to_output_distrs, subnet_to_flops, labels, if_allow_noop, ensemble_size, logit_gaps_all, **kwargs):
        super().__init__()
        # ConfidentModel is implemented as boolean idx atop model's predictions
        torch.multiprocessing.set_start_method('spawn')
        # need to set up logging in subprocesses, so:
        self.log_file_path = kwargs['log_file_path']

        self.subnet_to_output_distrs = torch.tensor(subnet_to_output_distrs)
        self.subnet_to_output_distrs_argmaxed = torch.argmax(self.subnet_to_output_distrs, -1)

        self.subnet_to_logit_gaps = torch.tensor(logit_gaps_all) + 1e-7 # float16 => some are zeroes, and that breaks the 0 threshold

        subnet_to_logit_gaps_fp32 = torch.tensor(self.subnet_to_logit_gaps, dtype=torch.float32)
        self.subnet_to_logit_gaps_unique_sorted = {}
        for i in range(len(self.subnet_to_logit_gaps)):
            tmp = torch.sort(torch.unique(subnet_to_logit_gaps_fp32[i]))[0]
            self.subnet_to_logit_gaps_unique_sorted[i] = tmp

        self.labels = torch.tensor(labels)
        self.subnet_to_flops = subnet_to_flops
        self.size_to_pad_to = ensemble_size # to pad to this size - doesn't influence the algorithm, only the ease of saving
        # In a bunch of places in the code I make the assumption that noop is included

    @staticmethod
    def acc_constraint_holds(preds1, preds2, labels, multiplier_ref_acc):
        def accuracy(preds, labels):
            return torch.sum(labels == preds) / len(labels) * 100
        acc1 = accuracy(preds1, labels)
        acc2 = accuracy(preds2, labels)
        if acc1 >= multiplier_ref_acc * acc2 and acc1 != 0:
            return acc1.item()
        return None

    @staticmethod
    def confident_model_set(validation_unpredicted_idx, idx_subnet_ref_acc, multiplier_ref_acc, already_used_subnets,
                            subnet_to_output_distrs_argmaxed, subnet_to_logit_gaps, subnet_to_logit_gaps_unique_sorted, labels):
        subnet_to_predicted_idx = {}
        subnet_to_threshold = {}
        subnet_to_acc = {}
        for i_model in range(subnet_to_output_distrs_argmaxed.shape[0]):
            if i_model in already_used_subnets:
                continue

            logit_gaps_cur = subnet_to_logit_gaps[i_model] # (n_samples, 1)

            logit_gaps_cur_unique_sorted = subnet_to_logit_gaps_unique_sorted[i_model]

            # first check zero, for speed
            t = 0.0
            predicted_idx = logit_gaps_cur >= t
            predicted_idx = predicted_idx * validation_unpredicted_idx  # only interested in predictions on yet-unpredicted images
            cur_subnet_to_output_distr_argmaxed = subnet_to_output_distrs_argmaxed[i_model]
            ref_subnet_to_output_distr_argmaxed = subnet_to_output_distrs_argmaxed[idx_subnet_ref_acc]
            acc = GreedySearchWrapperEnsembleClassification.acc_constraint_holds(cur_subnet_to_output_distr_argmaxed[predicted_idx],
                                            ref_subnet_to_output_distr_argmaxed[predicted_idx],
                                            labels[predicted_idx], multiplier_ref_acc)

            if acc is None:
                for ind in range(logit_gaps_cur_unique_sorted.shape[0]):
                    t = logit_gaps_cur_unique_sorted[ind]
                    predicted_idx = logit_gaps_cur >= t
                    predicted_idx = predicted_idx * validation_unpredicted_idx  # only interested in predictions on yet-unpredicted images

                    acc = GreedySearchWrapperEnsembleClassification.acc_constraint_holds(cur_subnet_to_output_distr_argmaxed[predicted_idx],
                                                    ref_subnet_to_output_distr_argmaxed[predicted_idx],
                                                    labels[predicted_idx], multiplier_ref_acc)
                    if acc is not None:
                        if ind > 0:
                            t = 1e-7 + logit_gaps_cur_unique_sorted[ind - 1].item()
                        else:
                            t = t.item()
                        break

            subnet_to_predicted_idx[i_model] = predicted_idx
            subnet_to_threshold[i_model] = t
            subnet_to_acc[i_model] = acc

        return subnet_to_predicted_idx, subnet_to_threshold, subnet_to_acc

    @staticmethod
    def _search_for_model_and_multiplier(kwargs):
        torch.set_num_threads(1)
        self, multiplier_ref_acc, idx_subnet_ref_acc = kwargs['self'], kwargs['multiplier_ref_acc'], kwargs['idx_subnet_ref_acc']
        utils.setup_logging(self.log_file_path)
        st = time.time()

        subnet_to_output_distrs = self.subnet_to_output_distrs
        subnet_to_output_distrs_argmaxed = self.subnet_to_output_distrs_argmaxed
        subnet_to_logit_gaps = self.subnet_to_logit_gaps
        labels = self.labels

        all_solutions = []
        all_objectives = []

        validation_unpredicted_idx = torch.ones_like(labels, dtype=bool)
        cur_cascade = [0]  # don't actually need noop, but helpful for saving (i.e. this is a crutch)
        cur_thresholds = []
        cur_flops = 0
        cur_predictions = torch.zeros_like(
            subnet_to_output_distrs[idx_subnet_ref_acc])  # idx is not important, they all have the same shape
        while torch.sum(validation_unpredicted_idx) > 0:
            subnet_to_predicted_idx, subnet_to_threshold, subnet_to_acc = \
                GreedySearchWrapperEnsembleClassification.confident_model_set(validation_unpredicted_idx,
                                                                              idx_subnet_ref_acc,
                                                                              multiplier_ref_acc, set(cur_cascade),
                                                                              subnet_to_output_distrs_argmaxed,
                                                                              subnet_to_logit_gaps, self.subnet_to_logit_gaps_unique_sorted, labels)

            best_new_subnet_index = -1
            best_r = 0

            for i_model in subnet_to_predicted_idx.keys():
                n_predicted_cur = torch.sum(subnet_to_predicted_idx[i_model])
                if n_predicted_cur == 0:
                    continue
                # filter to M_useful
                if subnet_to_acc[i_model] is None:
                    continue
                # the "confident_model_set", as described in the paper, already generates only models that satisfy
                # the accuracy constraint, thus M_useful and M_accurate are the same
                r = n_predicted_cur / self.subnet_to_flops[i_model]
                if r > best_r:
                    best_r = r
                    best_new_subnet_index = i_model

            cur_cascade.append(best_new_subnet_index)
            cur_thresholds.append(subnet_to_threshold[best_new_subnet_index])
            cur_flops += (torch.sum(validation_unpredicted_idx) / len(labels)) * self.subnet_to_flops[
                best_new_subnet_index]

            predicted_by_best_subnet_idx = subnet_to_predicted_idx[best_new_subnet_index]
            cur_predictions[predicted_by_best_subnet_idx] = subnet_to_output_distrs[best_new_subnet_index][
                predicted_by_best_subnet_idx]
            validation_unpredicted_idx[predicted_by_best_subnet_idx] = False

        cur_flops = cur_flops.item()

        cur_cascade = cur_cascade[1:] # I think this should work

        if len(cur_cascade) < self.size_to_pad_to:
            n_to_add = self.size_to_pad_to - len(cur_cascade)
            cur_cascade += [0] * n_to_add
            cur_thresholds += [0] * n_to_add
        all_solutions.append(cur_cascade + cur_thresholds)
        # compute true error for the constructed cascade
        output = torch.argmax(cur_predictions, 1)
        true_err = (torch.sum(labels != output) / len(labels) * 100).item()
        all_objectives.append((true_err, cur_flops))

        ed = time.time()
        print(f'{multiplier_ref_acc=} {idx_subnet_ref_acc=} {cur_cascade=} {cur_thresholds=} {cur_flops=:.2f} time={ed-st}')

        return all_solutions, all_objectives

    def search(self, seed):
        # Seed is not useful and not used because the algorithm is deterministic.
        st = time.time()
        all_solutions = []
        all_objectives = []

        kwargs = []
        for idx_subnet_ref_acc in range(1, len(self.subnet_to_output_distrs)):
                for multiplier_ref_acc in [1 - i / 100 for i in range(0, 5 + 1)]:
                    kwargs.append({'self': self, 'multiplier_ref_acc': multiplier_ref_acc,
                                   'idx_subnet_ref_acc': idx_subnet_ref_acc})

        n_workers = 32
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(GreedySearchWrapperEnsembleClassification._search_for_model_and_multiplier, kws) for kws in kwargs]
            for f in futures:
                cur_solutions, cur_objectives = f.result() # the order is not important
                all_solutions += cur_solutions
                all_objectives += cur_objectives

        all_solutions = np.vstack(all_solutions)
        all_objectives = np.array(all_objectives)
        print(all_solutions)
        print(all_objectives)
        ed = time.time()
        print(f'GreedyCascade time = {ed - st}')
        return all_solutions, all_objectives