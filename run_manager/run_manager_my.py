from collections import defaultdict
import time
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import torchvision
from ofa.utils import AverageMeter, accuracy


class RunManagerMy:
    def __init__(self, net, run_config, no_gpu=False, sec_obj='flops'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and (not no_gpu) else 'cpu')
        self.is_ensemble = isinstance(net, list)
        if self.is_ensemble:
            for net_i in net:
                net_i.to(self.device)
        else:
            net.to(self.device)

        self.accuracy = accuracy
        self.get_scalar_from_accuracy = lambda acc: acc[0].item()
        self.if_enough_vram = False
        self.sec_obj = sec_obj
        self.run_config = run_config
        self.test_criterion = nn.CrossEntropyLoss()

    def update_metric(self, metric_dict, output, labels):
        acc1 = self.accuracy(output, labels, topk=(1,))
        acc1 = self.get_scalar_from_accuracy(acc1)
        metric_dict['top1'].update(acc1, output.size(0))

    def validate(self, is_test=False, net=None, data_loader=None, no_logs=False,
                 if_return_outputs=False, resolutions_list=None, thresholds=None, if_return_logit_gaps=False,
                 if_use_logit_gaps=False):
        assert not(if_use_logit_gaps and if_return_logit_gaps) # they aren't really mutually exclusive, but it's simpler this way

        if if_return_outputs:
            outputs_to_return = []
        if if_return_logit_gaps:
            logit_gaps_to_return = []

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        if not self.is_ensemble:
            net.eval()
        else:
            for net_cur in net:
                net_cur.eval()

        losses = AverageMeter()
        metric_dict = defaultdict(lambda: AverageMeter())

        if_cascade = thresholds is not None
        if if_cascade:
            n_not_predicted_per_stage = [0 for _ in range(len(net) - 1)]

        with torch.no_grad(), torch.cuda.amp.autocast():
            with tqdm(total=len(data_loader), desc='Evaluate ', disable=no_logs) as t:
                st = time.time()
                for i, (images, labels, *_) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    images_orig = None  # don't make a backup unless I need to

                    if not self.is_ensemble:
                        output = net(images)
                    else:
                        out_logits = net[0](images)
                        output = torch.nn.functional.softmax(out_logits, dim=1)
                        if if_return_logit_gaps or if_use_logit_gaps:
                            # it only make sense to store the logit gaps if this is a separate network, not an ensemble/cascade
                            two_max_values = out_logits.topk(k=2, dim=-1).values
                            logit_gap = two_max_values[:, 0] - two_max_values[:, 1]

                        if if_cascade:
                            idx_more_predictions_needed = torch.ones(images.shape[0], dtype=torch.bool)

                        i_net = 1
                        for net_cur in net[1:]:
                            if if_cascade:
                                cur_threshold = thresholds[i_net - 1]
                                if if_use_logit_gaps:
                                    idx_more_predictions_needed[logit_gap >= cur_threshold] = False
                                else:
                                    idx_more_predictions_needed[torch.max(output, dim=1).values >= cur_threshold] = False
                                output_tmp = output[idx_more_predictions_needed]
                                if len(output_tmp) == 0:
                                    n_not_predicted = 0
                                else:
                                    if if_use_logit_gaps:
                                        logit_gap_tmp = logit_gap[idx_more_predictions_needed]
                                        not_predicted_idx = logit_gap_tmp < cur_threshold
                                    else:
                                        not_predicted_idx = torch.max(output_tmp, dim=1).values < cur_threshold
                                    n_not_predicted = torch.sum(not_predicted_idx).item()
                                n_not_predicted_per_stage[i_net - 1] += n_not_predicted
                                if n_not_predicted == 0:
                                    break

                            if resolutions_list is not None:
                                if resolutions_list[i_net] != resolutions_list[i_net - 1]:
                                    if images_orig is None:
                                        images_orig = torch.clone(images)
                                    r = resolutions_list[i_net]
                                    images = torchvision.transforms.functional.resize(images_orig, (r, r))

                            if not if_cascade:
                                output_cur = torch.nn.functional.softmax(net_cur(images), dim=1)
                            else:
                                out_logits = net_cur(images[idx_more_predictions_needed][not_predicted_idx])
                                if len(out_logits.shape) < 2: #a single image is left in the batch, need to fix dim
                                    out_logits = out_logits[None,...]
                                output_cur = torch.nn.functional.softmax(out_logits, dim=1)

                            if not if_cascade:
                                output += output_cur
                            else:
                                if if_use_logit_gaps:
                                    # firstly, need to overwrite previous predictions (because they didn't really happen if the gap was too small)
                                    output_tmp[not_predicted_idx] = output_cur
                                    output[idx_more_predictions_needed] = output_tmp

                                    # secondly, need to update the logit gap
                                    two_max_values = out_logits.topk(k=2, dim=-1).values
                                    logit_gap_tmp[not_predicted_idx] = two_max_values[:, 0] - two_max_values[:, 1]
                                    # note that the gap for the previously predicted values will be wrong, but it doesn't matter
                                    # because the idx for them has already been set to False
                                    logit_gap[idx_more_predictions_needed] = logit_gap_tmp
                                else:
                                    n_nets_used_in_cascade = i_net + 1
                                    coeff1 = ((n_nets_used_in_cascade - 1) / n_nets_used_in_cascade)
                                    coeff2 = (1 / n_nets_used_in_cascade)
                                    output_tmp[not_predicted_idx] = coeff1 * output_tmp[not_predicted_idx] \
                                                                 + coeff2 * output_cur #don't need idx here because had it for images passed to the net_cur
                                    output[idx_more_predictions_needed] = output_tmp

                            i_net += 1

                        if not if_cascade:
                            output /= len(net)

                    if if_return_outputs:
                        outputs_to_return.append(output.detach().cpu())
                    if if_return_logit_gaps:
                        logit_gaps_to_return.append(logit_gap.detach().cpu())
                    loss = self.test_criterion(output, labels)
                    self.update_metric(metric_dict, output, labels)

                    losses.update(loss.item(), images.size(0))
                    t.set_postfix({'loss': losses.avg,
                                    **self.get_metric_vals(metric_dict),
                                    'img_size': images.size(2)})
                    t.update(1)

                ed = time.time()
                print(f'Forward time {ed - st}')

        dict_to_return = self.get_metric_vals(metric_dict)
        if if_return_outputs:
            outputs_to_return = torch.cat(outputs_to_return, dim=0).numpy()
            dict_to_return['output_distr'] = outputs_to_return
        if if_return_logit_gaps:
            logit_gaps_to_return = torch.cat(logit_gaps_to_return, dim=0).numpy()
            dict_to_return['logit_gaps'] = logit_gaps_to_return
        if if_cascade:
            dict_to_return['n_not_predicted_per_stage'] = n_not_predicted_per_stage
        return losses.avg, dict_to_return

    def get_metric_vals(self, metric_dict):
        return {key: metric_dict[key].avg for key in metric_dict}