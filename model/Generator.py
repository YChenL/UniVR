import torch
import torch.nn as nn
import torch.nn.functional as F


def wn_convnx(n, *args, **kwargs):
    if n==1:
        conv = nn.Conv1d(*args, **kwargs)
    elif n==2:
        conv = nn.Conv2d(*args, **kwargs)
    elif n==3:
        conv = nn.Conv3d(*args, **kwargs)    
        
    nn.init.normal_(conv.weight, mean=0.0, std=0.02) 
    # nn.init.kaiming_normal_(conv.weight, nonlinearity='leaky_relu')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)
        
    return nn.utils.weight_norm(conv).cuda()


def wn_deconvnx(n, *args, **kwargs):
    if n==1:
        deconv = nn.ConvTranspose1d(*args, **kwargs)
    elif n==2:
        deconv = nn.ConvTranspose2d(*args, **kwargs)
    elif n==3:
        deconv = nn.ConvTranspose3d(*args, **kwargs)    
        
    nn.init.normal_(deconv.weight, mean=0.0, std=0.02) 
    # nn.init.kaiming_normal_(conv.weight, nonlinearity='leaky_relu')
    if deconv.bias is not None:
        nn.init.constant_(deconv.bias, 0)
        
    return nn.utils.weight_norm(deconv).cuda()


class Residual_block(nn.Module):
    def __init__(self, channels, dilation=1):
        super(Residual_block, self).__init__()
        # self.emb = nn.Conv1d
        self.dilation = dilation
        self.P = nn.ReflectionPad1d(dilation)
        self.conv_skip = wn_convnx(1, channels, channels, (1, ))
        self.conv1 = wn_convnx(1, channels, 2*channels, (3, ), dilation=(dilation, )) #kernel_size的第二个维度由in channel决定
        self.conv2 = wn_convnx(1, channels, channels, (1, ), dilation=(dilation, ))
        
    def forward(self, x, cond=None):
        # dim = x.size(1)
        skip = self.conv_skip(x)  
        x = self.P(x)
        x = self.conv1(x)           
        if cond is not None:
            x = x + self.emb(cond)
      
        a, b = x.split(x.size(1)//2, dim=1)
        x = torch.tanh(a) * torch.sigmoid(b)
        x = self.conv2(x)
        
        return x + skip
    

class Residual_stack(nn.Module):
    def __init__(self, hidden_layers, channels):
        super(Residual_stack, self).__init__()
        '''
        hidden_layers: int
        channels: int
        '''    
        self.stack = nn.Sequential()
        for i in range(hidden_layers):
            '''
            dilation = [1,3,9,27]
            '''
            self.stack.add_module('residual_blk'+str(i), Residual_block(channels, dilation=3**i))
        
    def forward(self, x, cond=None):
        x = self.stack(x)
        
        return x
    
    
class Encoder(nn.Module):
    r"""Implementation of the content encoder.
     residual_channels: list
     hidden_layer_num: list
     kernal_size: list 
     stride: list
     pad: list
    """
    def __init__(self, hidden_layer_num, residual_channels, kernal_size, stride, pad, d_con):
        super(Encoder, self).__init__()
        self.conv_top = wn_convnx(1, 1, 32, 7, 1, 3)
        self.conv_middle = nn.Sequential()
        for i in range(len(residual_channels)):
            self.conv_middle.add_module('Residual_stack'+str(i), Residual_stack(hidden_layer_num[i], residual_channels[i])) # residual_channels = [32, 64, 128, 256]
            self.conv_middle.add_module('Gelu'+str(i), nn.GELU())
            # self.conv_middle.add_module('Pad'+str(i), nn.ReflectionPad1d(pad[i]))
            self.conv_middle.add_module('Downsample'+str(i), wn_convnx(1, residual_channels[i], 2*residual_channels[i], kernal_size[i], stride[i], pad[i]))
   
        self.conv_bottom = nn.Sequential(nn.GELU(),
                                         nn.ReflectionPad1d(3),
                                         wn_convnx(1, 512, d_con, 7, 1),
                                         nn.GELU(),
                                         nn.ReflectionPad1d(3),
                                         wn_convnx(1, d_con, d_con, 7, 1))
               
            
    def forward(self, x):
        x = self.conv_top(x)
        x = self.conv_middle(x)
        x = self.conv_bottom(x)
        # Norm
        x = x / torch.sum(x**2 + 1e-12, dim=1, keepdim=True)**0.5
        return x
    
    
class Decoder(nn.Module):
    r"""Implementation of the content encoder.
     residual_channels: list
     hidden_layer_num: list
     kernal_size: list 
     stride: list
     pad: list
    """
    def __init__(self, hidden_layer_num, residual_channels, kernal_size, stride, pad, d_con):
        super(Decoder, self).__init__()
        self.conv_top = nn.Sequential(nn.ReflectionPad1d(3),
                                      wn_convnx(1, d_con, 512, 7, 1),
                                      nn.GELU(),
                                      nn.ReflectionPad1d(3),
                                      wn_convnx(1, 512, 512, 7, 1))
        
        self.conv_middle = nn.Sequential()
        for i in range(len(residual_channels)):   
            self.conv_middle.add_module('Gelu'+str(i), nn.GELU())
            self.conv_middle.add_module('Upsample'+str(i), wn_deconvnx(1, 2*residual_channels[i], residual_channels[i], kernal_size[i], stride[i], pad[i]))
            self.conv_middle.add_module('Residual_stack'+str(i), Residual_stack(hidden_layer_num[i], residual_channels[i])) # residual_channels = [32, 64, 128, 256]     
   
        self.conv_bottom = nn.Sequential(nn.GELU(),
                                         nn.ReflectionPad1d(3),
                                         wn_convnx(1, 32, 1, 7, 1),
                                         nn.Tanh())
               
            
    def forward(self, x):
        x = self.conv_top(x)
        x = self.conv_middle(x)
        x = self.conv_bottom(x)

        return x   
    
    
class ReconNet(nn.Module):
    r"""Implementation of the content encoder.
     residual_channels: list
     hidden_layer_num: list
     kernal_size: list 
     stride: list
     pad: list
    """
    def __init__(self, hidden_layer_num_d, residual_channels_d, kernal_size_d, stride_d, pad_d, 
                 hidden_layer_num_u, residual_channels_u, kernal_size_u, stride_u, pad_u, d_con):
        super(ReconNet, self).__init__()
        self.d_con = d_con
        self.conv_top = nn.Sequential(nn.ReflectionPad1d(3),
                                      wn_convnx(1, 1, 32, 7, 1))
                                      # nn.GELU(),
                                      # nn.ReflectionPad1d(3),
                                      # wn_convnx(1, 128, 512, 7, 1))
        
        self.conv_middle_down = nn.Sequential()
        for i in range(len(residual_channels_d)):
            self.conv_middle_down.add_module('Residual_stack'+str(i), Residual_stack(hidden_layer_num_d[i], residual_channels_d[i])) # residual_channels = [32, 64, 128, 256]
            self.conv_middle_down.add_module('Gelu'+str(i), nn.GELU())
            self.conv_middle_down.add_module('Downsample'+str(i), wn_convnx(1, residual_channels_d[i], 2*residual_channels_d[i], kernal_size_d[i], stride_d[i], pad_d[i]))      
            
        self.conv_middle_up = nn.Sequential()
        for i in range(len(hidden_layer_num_u)):
            self.conv_middle_up.add_module('Gelu'+str(i), nn.GELU())
            self.conv_middle_up.add_module('Upsample'+str(i), wn_deconvnx(1, 2*residual_channels_u[i], residual_channels_u[i], kernal_size_u[i], stride_u[i], pad_u[i]))
            self.conv_middle_up.add_module('Residual_stack'+str(i), Residual_stack(hidden_layer_num_u[i], residual_channels_u[i])) # residual_channels = [32, 64, 128, 256]     
            
        self.conv_bottom = nn.Sequential(nn.GELU(),
                                         nn.ReflectionPad1d(3),
                                         wn_convnx(1, 32, 1, 7, 1),
                                         nn.Tanh())
    def conv_bottleneck(self, inputs):
        
        x = nn.GELU()(inputs)
        x = nn.ReflectionPad1d(3)(x)
        x = wn_convnx(1, 512, self.d_con, 7, 1)(x)
        x = nn.GELU()(x)
        x = nn.ReflectionPad1d(3)(x) 
        emb = wn_convnx(1, self.d_con, self.d_con, 7, 1)(x)
        x = nn.GELU()(emb)
        x = nn.ReflectionPad1d(3)(x) 
        x = wn_convnx(1, self.d_con, 512, 7, 1)(x) 
        
        return x, emb

        # self.conv_bottleneck = nn.Sequential(nn.GELU(),
        #                                      nn.ReflectionPad1d(3),
        #                                      wn_convnx(1, 512, d_con, 7, 1),
        #                                      nn.GELU(),
        #                                      nn.ReflectionPad1d(3),
        #                                      wn_convnx(1, d_con, d_con, 7, 1), 
        #                                      nn.GELU(),   
        #                                      nn.ReflectionPad1d(3),
        #                                      wn_convnx(1, d_con, 512, 7, 1))
         
        
                         
    def forward(self, x):
        x = self.conv_top(x)
        x = self.conv_middle_down(x)
        x, emb = self.conv_bottleneck(x)
        x = self.conv_middle_up(x)
        x = self.conv_bottom(x)
        
        return x, emb