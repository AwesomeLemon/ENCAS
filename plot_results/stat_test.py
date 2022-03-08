from plotting_functions import *
from scipy.stats import wilcoxon

def get_wilcoxon_p(x, y):
    print(x)
    print(y)
    return wilcoxon(x, y, alternative='greater').pvalue

if __name__ == '__main__':
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    tmp_path = os.path.join(utils.NAT_PATH, '.tmp')

    print('0. ENCAS (1 supernetwork) > NAT (best)')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_1nets_sep_n5_evals600000_cascade_moregranular3_002',
        'cifar10_r0_attn_sep',], ['test', 'test_swa20'], 30,
                                  algo_names=['mo-gomea', 'search_algo:nsga3!subset_selector:reference'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_1nets_sep_n5_evals600000_cascade_moregranular3_002',
        'cifar100_r0_alpha_sep',], ['test', 'test_swa20'], 30,
                                  algo_names=['mo-gomea', 'search_algo:nsga3!subset_selector:reference'])
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_1nets_sep_n5_evals600000_cascade_moregranular3_002',
        'imagenet_r0_alpha_sep',], ['test', 'test'], 15,
                                  algo_names=['mo-gomea', 'search_algo:nsga3!subset_selector:reference'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0],
                                 all_hvs_c10[1] + all_hvs_c100[1] + all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0],
                                      all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))

    print('1. ENCAS (5 supernetworks) > NAT (best)')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'cifar10_r0_attn_sep',], ['test', 'test_swa20'], 30,
                                  algo_names=['mo-gomea', 'search_algo:nsga3!subset_selector:reference'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'cifar100_r0_alpha_sep',], ['test', 'test_swa20'], 30,
                                  algo_names=['mo-gomea', 'search_algo:nsga3!subset_selector:reference'])
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'imagenet_r0_alpha_sep',], ['test', 'test'], 15,
                                  algo_names=['mo-gomea', 'search_algo:nsga3!subset_selector:reference'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0],
                                 all_hvs_c10[1] + all_hvs_c100[1] + all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0],
                                      all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))


    print('2. ENCAS (5 supernetworks) > ENCAS (1 supernetwork)')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_1nets_sep_n5_evals600000_cascade_moregranular3_002',
        ], ['test'] * 2, 30, algo_names=['mo-gomea']*2)
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_1nets_sep_n5_evals600000_cascade_moregranular3_002',
        ], ['test'] * 2, 30, algo_names=['mo-gomea'] * 2)
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_imagenet_r0_1nets_sep_n5_evals600000_cascade_moregranular3_002',
        ], ['test', 'test'], 15,
                                  algo_names=['mo-gomea'] * 2)
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0], all_hvs_c10[1] + all_hvs_c100[1] + all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0], all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))

    print('3. ENCAS (5 supernetworks) > GreedyCascade')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3',], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'greedy'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3',], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'greedy'])
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3',], ['test', 'test'], 15,
                                  algo_names=['mo-gomea', 'greedy'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0],
                                 all_hvs_c10[1] + all_hvs_c100[1] + all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0],
                                      all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))

    print('4. ENCAS-joint+ > ENCAS-joint')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_join_n5_evals600000_cascade_moregranular3_002',
        'cifar10_r0_5nets'], ['test', 'test_swa20'], 30,
                                  algo_names=['mo-gomea', 'search_algo:mo-gomea!subset_selector:reference'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_join_n5_evals600000_cascade_moregranular3_002',
        'cifar100_r0_5nets',], ['test', 'test_swa20'], 30,
                                  algo_names=['mo-gomea', 'search_algo:mo-gomea!subset_selector:reference'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0], all_hvs_c10[1] + all_hvs_c100[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0], all_max_accs_c10[1] + all_max_accs_c100[1]))

    print('5. ENCAS-joint+ > ENCAS')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_join_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002'], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_join_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0], all_hvs_c10[1] + all_hvs_c100[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0], all_max_accs_c10[1] + all_max_accs_c100[1]))

    print('6. ENCAS (5 supernetworks) > ENCAS with 5 clones of the best supernet')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_clones_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    ], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_clones_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    ], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0], all_hvs_c10[1] + all_hvs_c100[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0], all_max_accs_c10[1] + all_max_accs_c100[1]))

    print('7. ENCAS + MO-GOMEA > ENCAS + Random search (val)')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    ], ['val', 'val'], 30,
                                  algo_names=['mo-gomea', 'random'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    ], ['val', 'val'], 30,
                                  algo_names=['mo-gomea', 'random'])
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002'], ['val', 'val'], 15,
                                  algo_names=['mo-gomea', 'random'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0], all_hvs_c10[1] + all_hvs_c100[1]+ all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0], all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))

    print('8. ENCAS + MO-GOMEA > ENCAS + Random search (test)')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    ], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'random'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    ], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'random'])
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002'], ['test', 'test'], 15,
                                  algo_names=['mo-gomea', 'random'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0], all_hvs_c10[1] + all_hvs_c100[1]+ all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0], all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))

    print('9. ENCAS (5 supernetworks) > ENCAS-ensemble (5 supernetworks)')
    all_hvs_c10, all_max_accs_c10 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar10_r0_swa20_5nets_sep_n5_evals600000_ensemble_moregranular3',
    ], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    all_hvs_c100, all_max_accs_c100 = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_ensemble_moregranular3',
    ], ['test', 'test'], 30,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    all_hvs_img, all_max_accs_img = get_hypervolumes_and_max_accs_for_stat_testing([
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
        'posthoc_imagenet_r0_5nets_sep_n5_evals600000_ensemble_moregranular3'], ['test', 'test'], 15,
                                  algo_names=['mo-gomea', 'mo-gomea'])
    print('hv: ', get_wilcoxon_p(all_hvs_c10[0] + all_hvs_c100[0] + all_hvs_img[0], all_hvs_c10[1] + all_hvs_c100[1] + all_hvs_img[1]))
    print('max acc: ', get_wilcoxon_p(all_max_accs_c10[0] + all_max_accs_c100[0] + all_max_accs_img[0], all_max_accs_c10[1] + all_max_accs_c100[1] + all_max_accs_img[1]))
