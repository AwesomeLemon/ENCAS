# taken from https://github.com/facebookresearch/AttentiveNAS
# Difference: images not resized in forward, but beforehand, in the collator (which is faster)

import torch
import torch.nn as nn
from .modules_alphanet.nn_base import MyNetwork

class AttentiveNasStaticModel(MyNetwork):

    def __init__(self, first_conv, blocks, last_conv, classifier, use_v3_head=True):
        super(AttentiveNasStaticModel, self).__init__()
        
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.last_conv = last_conv
        self.classifier = classifier

        self.use_v3_head = use_v3_head

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        if not self.use_v3_head:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str = '\n'.join([block.module_str for block in self.blocks])
        #_str += self.last_conv.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': AttentiveNasStaticModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            #'last_conv': self.last_conv.config,
            'classifier': self.classifier.config,
            # 'resolution': self.resolution
        }


    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None # cumulative moving average
                m.reset_running_stats()



