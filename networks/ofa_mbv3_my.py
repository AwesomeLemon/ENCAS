import copy

import torch
from ofa.imagenet_classification.elastic_nn.modules import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.networks import MobileNetV3
from ofa.utils import val2list, make_divisible, MyNetwork
from ofa.utils.layers import ConvLayer, MBConvLayer, ResidualBlock, IdentityLayer, LinearLayer, My2DLayer
from torch.utils.checkpoint import checkpoint_sequential

class OFAMobileNetV3My(OFAMobileNetV3):
    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None, width_mult=1.0,
                 ks_list=3, expand_ratio_list=6, depth_list=4, if_use_gradient_checkpointing=False,
                 class_for_subnet=MobileNetV3, n_image_channels=3):
        '''
        differences to init of super:
        1) several widths in each block instead of one => unneccessary, since NAT turned out to use 2 separate supernets
        2) arbitrary n_image_channels => not used on cifars or imagenet
        3) specify class_for_subnet => not used in classification
        '''
        self.width_mult = val2list(width_mult)
        # self.width_mults = [1.0, 1.2]
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.n_image_channels = n_image_channels

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = [make_divisible(base_stage_width[-2] * w, MyNetwork.CHANNEL_DIVISIBLE) for w in
                              self.width_mult]
        last_channel = [make_divisible(base_stage_width[-1] * w, MyNetwork.CHANNEL_DIVISIBLE) for w in self.width_mult]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_lists = []
        for base_width in base_stage_width[:-2]:
            width_list_cur = []
            for w in self.width_mult:
                width = make_divisible(base_width * w, MyNetwork.CHANNEL_DIVISIBLE)
                width_list_cur.append(width)
            width_lists.append(width_list_cur)

        input_channel, first_block_dim = width_lists[0], width_lists[1]
        # first conv layer
        first_conv = DynamicConvLayer([self.n_image_channels], input_channel, kernel_size=3, stride=2, act_func='h_swish')
        first_block_conv = DynamicMBConvLayer(
            in_channel_list=input_channel, out_channel_list=first_block_dim, kernel_size_list=3,
            stride=stride_stages[0],
            expand_ratio_list=1, act_func=act_stages[0], use_se=se_stages[0],
        )
        first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(max(first_block_dim), max(first_block_dim)) if input_channel == first_block_dim else None,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = first_block_dim

        for width_list_cur, n_block, s, act_func, use_se in zip(width_lists[2:], n_block_list[1:],
                                                                stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width_list_cur
            for i in range(n_block):
                stride = s if i == 0 else 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(width_list_cur),
                    kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(max(feature_dim), max(feature_dim))
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        # final expand layer, feature mix layer & classifier
        final_expand_layer = DynamicConvLayer(feature_dim, final_expand_width, kernel_size=1, act_func='h_swish')
        feature_mix_layer = DynamicConvLayer(
            final_expand_width, last_channel, kernel_size=1, use_bn=False, act_func='h_swish',
        )

        classifier = DynamicLinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(OFAMobileNetV3, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.if_use_gradient_checkpointing = if_use_gradient_checkpointing
        self.class_for_subnet = class_for_subnet


    def set_active_subnet(self, ks=None, e=None, d=None, w=None, **kwargs):
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))
        width_mult = 0# since it turned out that different widths <=> different supernets, there's always just one width_mult, so we just take that.

        self.first_conv.active_out_channel = self.first_conv.out_channel_list[width_mult]
        self.blocks[0].conv.active_out_channel = self.blocks[0].conv.out_channel_list[width_mult]
        self.final_expand_layer.active_out_channel = self.final_expand_layer.out_channel_list[width_mult]
        self.feature_mix_layer.active_out_channel = self.feature_mix_layer.out_channel_list[width_mult]

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e
            block.conv.active_out_channel = block.conv.out_channel_list[width_mult]

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def get_active_subnet(self, preserve_weight=True):
        first_conv = self.first_conv.get_active_subnet(self.n_image_channels, preserve_weight)
        blocks = [
            ResidualBlock(self.blocks[0].conv.get_active_subnet(self.first_conv.active_out_channel),
                          copy.deepcopy(self.blocks[0].shortcut))
        ]

        final_expand_layer = self.final_expand_layer.get_active_subnet(self.blocks[-1].conv.active_out_channel)
        feature_mix_layer = self.feature_mix_layer.get_active_subnet(self.final_expand_layer.active_out_channel)
        classifier = self.classifier.get_active_subnet(self.feature_mix_layer.active_out_channel)

        input_channel = blocks[0].conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.blocks[idx].shortcut)
                ))
                input_channel = self.blocks[idx].conv.active_out_channel
            blocks += stage_blocks

        _subnet = self.class_for_subnet(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def forward(self, x):
        if not (self.if_use_gradient_checkpointing and self.training):
            # first conv
            x = self.first_conv(x)
            # first block
            x = self.blocks[0](x)
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                for idx in active_idx:
                    x = self.blocks[idx](x)
            x = self.final_expand_layer(x)
        else:
            x = self.first_conv(x)
            blocks_to_run = [self.blocks[0]]
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                for idx in active_idx:
                    blocks_to_run.append(self.blocks[idx])
            blocks_to_run.append(self.final_expand_layer)
            x = checkpoint_sequential(blocks_to_run, 2, x)

        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
