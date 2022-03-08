from plot_results.plotting_functions import compare_val_and_test
from after_search.evaluate_stored_outputs import evaluate_stored_whole_experiment
from after_search.extract_supernet_from_joint import extract_all
from after_search.store_outputs import store_cumulative_pareto_front_outputs


def extract_store_eval(dataset, exp_name, supernets, swa, **kwargs):
    '''
    A single function for ENCAS-joint that creates per-supernetwork trade-off fronts, evaluates them,
    and stores the outputs.
    '''
    extract_all(exp_name, supernets, swa=swa)
    max_iter = kwargs.get('max_iter', 30)
    target_runs = kwargs.get('target_runs', None)
    dataset_to_label_path = {'cifar100': 'labels_cifar100_test.npy', 'cifar10': 'labels_cifar10_test.npy',
                             'imagenet': 'labels_imagenet_test.npy'}

    for i, out_name_suffix in enumerate(supernets):
        out_name = exp_name + f'_EXTRACT_{out_name_suffix}'
        store_cumulative_pareto_front_outputs(out_name, 'val', max_iter=max_iter, swa=swa, target_runs=target_runs)
        store_cumulative_pareto_front_outputs(out_name, 'test', max_iter=max_iter, swa=swa, target_runs=target_runs)
        evaluate_stored_whole_experiment(out_name, f'test_swa{swa}', dataset_to_label_path[dataset],
                                         max_iter=max_iter, target_runs=target_runs)
        compare_val_and_test(out_name, f'test_swa{swa}', max_iter=max_iter, target_runs=target_runs)

if __name__=='__main__':
    extract_store_eval('cifar100', 'cifar100_r0_5nets', ['alpha', 'ofa12', 'ofa10', 'attn', 'proxyless'], 20, max_iter=30)
