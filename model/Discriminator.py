import torch
import torch.nn as nn
from .blocks import LinearNorm, ConvNorm, DiffusionEmbedding, Mish
import torch.nn.functional as F

class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, preprocess_config, model_config, train_config):
        super(JCUDiscriminator, self).__init__()

        n_mel_channels = 128 # preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        n_layer = model_config["discriminator"]["n_layer"]
        n_uncond_layer = model_config["discriminator"]["n_uncond_layer"]
        n_cond_layer = model_config["discriminator"]["n_cond_layer"]
        n_channels = model_config["discriminator"]["n_channels"]
        kernel_sizes = model_config["discriminator"]["kernel_sizes"]
        strides = model_config["discriminator"]["strides"]
        self.multi_speaker = model_config["multi_speaker"]  #True

        self.input_projection = LinearNorm(n_mel_channels, 2 * n_mel_channels) #输入是（real_x, fake_x) concat的，所以输入通道翻倍
        #self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        )
        if self.multi_speaker:
            self.spk_mlp = nn.Sequential(
                LinearNorm(residual_channels, n_channels[n_layer-1]),
            )
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_mel_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer)
            ]
        )
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts): #, x_t_prevs, s
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        """
        x = self.input_projection(x_ts).transpose(1, 2)
                   
        #torch.cat([x_t_prevs, ], dim=-1)
        # diffusion_step = self.mlp(self.diffusion_embedding(t)).unsqueeze(-1)
        if self.multi_speaker:
            speaker_emb = self.spk_mlp(s).unsqueeze(-1)

        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        # x_cond = (x + diffusion_step + speaker_emb) \
        #     if self.multi_speaker else (x + diffusion_step)     
        x_cond = (x + speaker_emb) \
            if self.multi_speaker else x
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats