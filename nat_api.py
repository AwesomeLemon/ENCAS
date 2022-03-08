import pickle

import numpy as np

from networks.attentive_nas_dynamic_model import AttentiveNasDynamicModel
from networks.ofa_mbv3_my import OFAMobileNetV3My
from networks.proxyless_my import OFAProxylessNASNetsMy

from search_space.ensemble_ss import EnsembleSearchSpace
from utils import get_metric_complement, get_net_info, SupernetworkWrapper


class NatAPI():
    def __init__(self, filename):
        super().__init__()
        self.use_cache = True

        # some variables are unused, kept for backwards compatibility
        predictor, n_classes, supernet_paths, archive_path, sec_obj, _, alphabet_name, n_image_channels, dataset, \
            search_space_name, ensemble_ss_names, _ = pickle.load(open(filename, 'rb'))

        self.search_space = EnsembleSearchSpace(ensemble_ss_names,
                                                [{'alphabet':alphabet_name_cur, 'ensemble_size': len(alphabet_name)}
                                                 for alphabet_name_cur in alphabet_name])
        self.predictor = predictor
        self.sec_obj = sec_obj
        self.n_image_channels = n_image_channels
        if search_space_name == 'ensemble':
            # assume supernet_paths is a list of paths, 1 per supernet
            ss_name_to_class = {'alphanet': AttentiveNasDynamicModel, 'ofa': OFAMobileNetV3My,
                                    'proxyless': OFAProxylessNASNetsMy}

            classes_to_use = [ss_name_to_class[ss_name] for ss_name in ensemble_ss_names]
            self.evaluators = [SupernetworkWrapper(n_classes=n_classes, model_path=supernet_path,
                                                   engine_class_to_use=encoder_class,
                                                   n_image_channels=self.n_image_channels, if_ignore_decoder=False, dataset=dataset,
                                                   search_space_name=ss_name, decoder_name='')
                               for supernet_path, ss_name, encoder_class in zip(supernet_paths, ensemble_ss_names, classes_to_use)]


    def fitness(self, solution):
        solution = [int(x) for x in solution]
        config = self.search_space.decode(solution)

        sec_objs = []
        for conf, evaluator in zip(config, self.evaluators):
            subnet, _ = evaluator.sample({'ks': conf['ks'], 'e': conf['e'], 'd': conf['d'], 'w': conf['w']})
            info = get_net_info(subnet, (self.n_image_channels, conf['r'], conf['r']),
                                measure_latency=self.sec_obj, print_info=False, clean=True)
            sec_objs.append(info[self.sec_obj])
        if 'position' not in conf:
            obj1_proper_form = -sum(sec_objs)
            top1_err = self.predictor.predict(np.array(solution)[np.newaxis, :])[0]
            obj0_proper_form = get_metric_complement(top1_err[0])
        else:
            input_acc = np.array(solution)[np.newaxis, :]
            solution_reencoded_sep = self.search_space.encode(config, if_return_separate=True)
            input_flops = np.concatenate([sol_sep[-2:] for sol_sep in solution_reencoded_sep] + [[int(f) for f in sec_objs]])[np.newaxis, :]

            top1_err = self.predictor.predict({'for_acc': input_acc, 'for_flops': input_flops})[0]
            obj0_proper_form = get_metric_complement(top1_err[0])
            obj1_proper_form = -top1_err[1]

        # third objective was removed during code clean-up, but want to return 3 values for backward compatibility
        return (obj0_proper_form, obj1_proper_form, 0)