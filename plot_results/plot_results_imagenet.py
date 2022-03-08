from plotting_functions import *

if __name__ == '__main__':
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    # plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.grid'] = True

    from cycler import cycler

    plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#52bde0', '#FBC15E', '#8EBA42', '#FFB5B8', '#777777', 'tab:brown'])

    # Fig. 4
    # compare_test_many_experiments([
    #     'imagenet_r0_proxyless_sep',
    #     'imagenet_r0_ofa10_sep',
    #     'imagenet_r0_ofa12_sep',
    #     'imagenet_r0_attn_sep',
    #     'imagenet_r0_alpha_sep',
    #     # 'logs_classification/imagenet_NAT_front',
    # ], ['test'] * 5, if_plot_many_in_one=True, max_iters=15, out_name='baselines_imagenet', pdf=True)

    # Fig. 5
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#FBC15E', '#988ED5', '#348ABD'])
    # compare_test_many_experiments(['imagenet_r0_alpha_sep',
    #                                'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3',
    #                                'posthoc_imagenet_r0_1nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                # 'logs_classification/imagenet_efficientnet_front',
    #                                # 'logs_classification/imagenet_efficientnet_cascade_front',
    #                                ], ['test'] * 4, if_plot_many_in_one=True, max_iters=15,
    #                               algo_names=['search_algo:nsga3!subset_selector:reference', 'greedy', 'mo-gomea', 'mo-gomea',
    #                                           # 'effnet', 'effnet-cascade'
    #                                           ],
    #                               out_name='cascades_and_greedy_imagenet', pdf=True,
    #                               legend_labels=['NAT (best)', 'GreedyCascade', 'ENCAS (1 supernet)', 'ENCAS (5 supernets)'])
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#52bde0', '#FBC15E', '#8EBA42', '#FFB5B8', '#777777', 'tab:brown'])

    # Fig. 6
    # paths = ['posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #          'logs_classification/imagenet_efficientnet_front',
    #          'logs_classification/imagenet_NsgaNet2_front',
    #          'logs_classification/imagenet_alphanet_front',
    #          'logs_classification/imagenet_efficientnet_cascade_front',
    #          'logs_classification/imagenet_NEAS',
    #          'logs_classification/imagenet_MobileNetV3',
    #          'logs_classification/imagenet_BigNAS',
    #          'logs_classification/imagenet_OFA',
    #          ]
    # compare_test_many_experiments(paths, ['test'] * len(paths), if_plot_many_in_one=True, max_iters=15,
    #                               algo_names=['mo-gomea'] + ['whatever'] * (len(paths) - 1),
    #                               out_name='cmp_sota_imagenet', pdf=True,
    #                               legend_labels=['ENCAS', 'EfficientNet', 'NSGANetV2', 'AlphaNet', 'EfficientNet Cascade',
    #                                              'NEAS', 'MobileNetV3', 'BigNAS', 'OFA (#75)'])

    # Fig. 8
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#FBC15E'])
    # compare_test_many_experiments(['posthoc_imagenet_timm_1nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                 'logs_classification/timm_all',
    #                                'logs_classification/imagenet_efficientnet_front_full',
    #                                'logs_classification/imagenet_efficientnet_cascade_front_full',
    #
    #                                ],'test', algo_names=['mo-gomea', 'timm: trade-off front', 'EfficientNet', 'EfficientNet Cascade'],
    #                               legend_labels=['ENCAS', 'timm: trade-off front', 'EfficientNet', 'EfficientNet Cascade'],
    #                               max_iters=15, if_plot_many_in_one=True, out_name='timm', pdf=True,
    #                               if_log_scale_x=True)
    # plt.rcParams['axes.prop_cycle'] = cycler(color=['#E24A33', '#348ABD', '#988ED5', '#52bde0', '#FBC15E', '#8EBA42', '#FFB5B8', '#777777', 'tab:brown'])

    # Fig. 12
    # compare_test_many_experiments([
    #                                'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                ], ['test'] * 2, if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['random', 'mo-gomea'],# 'nsganetv2'],
    #                               out_name='random_acc_imagenet', pdf=True,
    #                               legend_labels=['Random', 'MO-GOMEA'])

    # Fig. 11
    # compare_test_many_experiments([
    #                                'posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002',
    #                                'posthoc_imagenet_r0_5nets_sep_n5_evals600000_ensemble_moregranular3',
    #                                ], ['test'] * 2, if_plot_many_in_one=True, max_iters=30,
    #                               algo_names=['mo-gomea'] * 2,# 'nsganetv2'],
    #                               out_name='ensemble_acc_imagenet', pdf=True,
    #                               legend_labels=['ENCAS', 'ENENS'], if_log_scale_x=True)

    # wanna know the median run to get the named models from it
    # compare_test_many_experiments(['posthoc_imagenet_r0_5nets_sep_n5_evals600000_cascade_moregranular3_002'],
    #                               'test', if_plot_many_in_one=True, max_iters=15,
    #                               algo_names=['mo-gomea'], print_median_run_flops_and_accs=True)