from ofa.imagenet_classification.elastic_nn.networks import OFAProxylessNASNets
from ofa.imagenet_classification.networks import ProxylessNASNets
import copy

from ofa.imagenet_classification.elastic_nn.modules import DynamicMBConvLayer
from ofa.utils import val2list, make_divisible, MyNetwork
from ofa.utils.layers import ConvLayer, MBConvLayer, ResidualBlock, IdentityLayer, LinearLayer


class OFAProxylessNASNetsMy(OFAProxylessNASNets):
    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None, width_mult=1.0,
                 ks_list=3, expand_ratio_list=6, depth_list=4, if_use_gradient_checkpointing=False,
                 class_for_subnet=ProxylessNASNets, n_image_channels=3):

        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        self.n_image_channels = n_image_channels
        self.class_for_subnet = class_for_subnet

        if base_stage_width == 'google':
            # MobileNetV2 Stage Width
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        else:
            # ProxylessNAS Stage Width
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]

        input_channel = make_divisible(base_stage_width[0] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        first_block_width = make_divisible(base_stage_width[1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        # first conv layer
        first_conv = ConvLayer(
            self.n_image_channels, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        # first block
        first_block_conv = MBConvLayer(
            in_channels=input_channel, out_channels=first_block_width, kernel_size=3, stride=1,
            expand_ratio=1, act_func='relu6',
        )
        first_block = ResidualBlock(first_block_conv, None)

        input_channel = first_block_width
        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 1, 2, 1]
        n_block_list = [max(self.depth_list)] * 5 + [1]

        width_list = []
        for base_width in base_stage_width[2:-1]:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)

        for width, n_block, s in zip(width_list, n_block_list, stride_stages):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(input_channel, 1), out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list, stride=stride, act_func='relu6',
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = ResidualBlock(mobile_inverted_conv, shortcut)

                blocks.append(mb_inverted_block)
                input_channel = output_channel
        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6',
        )
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(OFAProxylessNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.width_mult = [self.width_mult]

    def get_active_subnet(self, preserve_weight=True):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

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
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = self.class_for_subnet(first_conv, blocks, feature_mix_layer, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet