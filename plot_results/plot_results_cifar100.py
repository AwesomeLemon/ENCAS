from plotting_functions import *

if __name__ == '__main__':
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    # plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.grid'] = True

    tmp_path = os.path.join(utils.NAT_PATH, '.tmp')
    from cycler import cycler
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', 'c', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8', 'tab:brown'])
    plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#52bde0', '#FBC15E', '#8EBA42', '#FFB5B8', '#777777', 'tab:brown'])

    # Fig. 4
    # plt.rcParams.update({'font.size': 16})
    # compare_test_many_experiments([
    #     'cifar100_r0_proxyless_sep',
    #     'cifar100_r0_ofa10_sep',
    #     'cifar100_r0_ofa12_sep',
    #     'cifar100_r0_attn_sep',
    #     'cifar100_r0_alpha_sep',
    #     'cifar100_reproducenat',
    # ], ['test_swa20']*5 + ['test'], if_plot_many_in_one=True, max_iters=30,
    #     out_name='baselines_cifar100', pdf=True)
    # plt.rcParams.update({'font.size': 18})

    # Fig. 5
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#FBC15E', '#988ED5', '#348ABD'])
    # compare_test_many_experiments(['cifar100_r0_alpha_sep',
    #                                'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3',
    #                                'posthoc_cifar100_r0_swa20_1nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                # 'logs_classification/cifar100_NsgaNet2_front',
    #                                ], ['test_swa20'] + ['test'] * 3, if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['search_algo:nsga3!subset_selector:reference', 'greedy', 'mo-gomea', 'mo-gomea'],# 'nsganetv2'],
    #                               out_name='cascades_and_greedy_cifar100', pdf=True,
    #                               legend_labels=['NAT (best)', 'GreedyCascade', 'ENCAS (1 supernet)',
    #                                              'ENCAS (5 supernets)'])
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#52bde0', '#FBC15E', '#8EBA42', '#FFB5B8', '#777777', 'tab:brown'])

    # Fig. 6
    # compare_test_many_experiments(['posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'logs_classification/cifar100_efficientnet_front',
    #                                'logs_classification/cifar100_NsgaNet2_front',
    #                                'logs_classification/cifar100_GDAS',
    #                                'logs_classification/cifar100_SETN',
    #                                ], ['test'] * 5, if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['mo-gomea'] + ['whatever'] * 4,
    #                               out_name='cmp_sota_cifar100', pdf=True,
    #                               legend_labels=['ENCAS', 'EfficientNet', 'NSGANetV2', 'GDAS', 'SETN'])

    # Fig. 7
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33','#348ABD', '#8EBA42',  '#FBC15E'])
    # compare_test_many_experiments(['cifar100_r0_attn_sep',
    #                                 'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'cifar100_r0_5nets',
    #                                'posthoc_cifar100_r0_swa20_5nets_join_n5_evals600000_cascade_moregranular3_002',
    #                                ], ['test_swa20', 'test', 'test_swa20', 'test'], if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['search_algo:nsga3!subset_selector:reference','mo-gomea',
    #                                           'search_algo:mo-gomea!subset_selector:reference', 'mo-gomea'],
    #                               out_name='sep_vs_join_cifar100', pdf=True,
    #                               legend_labels=['NAT (best)','ENCAS', 'ENCAS-joint', 'ENCAS-joint+'],# target_runs=[0, 1],
    #                               if_log_scale_x=True)
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#52bde0', '#FBC15E', '#8EBA42', '#FFB5B8', '#777777', 'tab:brown'])

    # Fig. 9: HVs: impact_n_supernets
    # plot_hypervolumes_impact_n_supernets([
    #     'posthoc_cifar100_r0_swa20_1nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_2nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                          ], 'test',
    #                         30, ['mo-gomea']  * 3, supernet_numbers=[1, 2, 5], set_xticks=True, label='ENCAS')#, target_runs=[0, 1])
    # plot_hypervolumes_impact_n_supernets([
    #     'cifar100_r0_alphaofa',
    #     'cifar100_r0_5nets'], 'test_swa20',
    #                         30, ['search_algo:mo-gomea!subset_selector:reference']  * 2, supernet_numbers=[2, 5],
    #     set_xticks=False, label='ENCAS-joint')
    # plot_hypervolumes_impact_n_supernets([
    #     'posthoc_cifar100_r0_swa20_2nets_join_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_5nets_join_n5_evals600000_cascade_moregranular3_002'], 'test',
    #                         30, ['mo-gomea']  * 2, supernet_numbers=[2, 5],
    #     set_xticks=False, label='ENCAS-joint+')
    # plt.legend()
    # plt.xlabel('Number of supernetworks')
    # plt.ylabel('Hypervolume')
    # plt.savefig(os.path.join(tmp_path, 'impact_n_supernets_cifar100.pdf'), bbox_inches='tight', pad_inches=0.01)
    # plt.close()

    # Fig. 10: HVs: impact_n_clones
    # plt.figure(figsize=(8, 4))
    # plot_hypervolumes_impact_n_supernets([
    #     'posthoc_cifar100_r0_swa20_2nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_3nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_4nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                          ], 'test',
    #                         30, ['mo-gomea']  * 4, supernet_numbers=[2, 3, 4, 5], set_xticks=True, label='Different supernets')
    # plot_hypervolumes_impact_n_supernets([
    #     'posthoc_cifar100_r0_swa20_clones_2nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_clones_3nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_clones_4nets_sep_n5_evals600000_cascade_moregranular3_002',
    #     'posthoc_cifar100_r0_swa20_clones_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                          ], 'test',
    #                         30, ['mo-gomea']  * 4, supernet_numbers=[2, 3, 4, 5], set_xticks=False, label='Different seeds')
    # plt.legend()
    # plt.xlabel('Number of supernetworks')
    # plt.ylabel('Hypervolume')
    # plt.savefig(os.path.join(tmp_path, 'impact_n_clones_cifar100.pdf'), bbox_inches='tight', pad_inches=0.01)
    # plt.close()

    # Fig. 12
    # compare_test_many_experiments([
    #                                'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                ], ['test'] * 2, if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['random', 'mo-gomea'],# 'nsganetv2'],
    #                               out_name='random_acc_cifar100', pdf=True,
    #                               legend_labels=['Random', 'MO-GOMEA'])
    #                               ,target_runs=[0, 1, 2, 3])

    # Fig. 11
    # compare_test_many_experiments([
    #                                'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_ensemble_moregranular3',
    #                                ], ['test'] * 2, if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['mo-gomea'] * 2,# 'nsganetv2'],
    #                               out_name='ensemble_acc_cifar100', pdf=True,
    #                               legend_labels=['ENCAS', 'ENENS'], if_log_scale_x=True)

    # wanna know the median run to get the named models from it
    # compare_test_many_experiments(['posthoc_cifar100_r0_swa20_5nets_sep_n5_evals600000_cascade_moregranular3_002'],
    #                               'test', if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['mo-gomea'], print_median_run_flops_and_accs=True)