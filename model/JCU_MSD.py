import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x
  

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal
   


class JCUD(nn.Module):
    def __init__(self):
        super(JCUD, self).__init__()
        '''
        '''
        # self.input_proj = LinearNorm(128, 256)
        self.conv_group = nn.ModuleList([    
            # nn.ReflectionPad1d(7),
            # nn.Sequential(
            #     ConvNorm(128, 256, kernel_size=7, stride=1, padding=1),
            #     nn.LeakyReLU(0.2, inplace=True),
            # ),
            nn.Sequential(
                ConvNorm(256, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                ConvNorm(64, 128, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),    
            nn.Sequential(
                ConvNorm(128, 512, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),    
            ),
            nn.Sequential(     
                ConvNorm(512, 128, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),     
            ConvNorm(128, 1, kernel_size=3, stride=1, padding=1),
                ])
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)
        
    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.conv_group:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]

    
    
class JCU_MSD(nn.Module):
    def __init__(self):
        super(JCU_MSD, self).__init__()        
        self.input_proj = LinearNorm(128, 256)
        self.discriminators = nn.ModuleList(
            [JCUD() for _ in range(3)]
        )
        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )
        
    def forward(self, x):
        ret = list()
        x = self.input_proj(x).transpose(1, 2)
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))

        return ret # [(feat, score), (feat, score), (feat, score)]