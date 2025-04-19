import imp
from re import L
from turtle import forward
import torch  # copy magnet1
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


########################################
class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



class FreqDec(nn.Module):
    '''Frequency Decomposition'''
    def __init__(self, channel_num):
        super(FreqDec, self).__init__()
        self.pwconv = nn.Conv2d(channel_num, channel_num*2, kernel_size=1)
        self.dwconv = nn.Conv2d(channel_num*2, channel_num*2, kernel_size=3, stride=1, padding=1, groups=channel_num*2)
        self.dila1 = nn.Conv2d(channel_num, channel_num, groups=channel_num, kernel_size=3, stride=1, padding=1)
        self.dila2 =nn.Conv2d(channel_num, channel_num, groups=channel_num, kernel_size=3, stride=1, padding=2, dilation = 2)
        self.R = nn.GELU()

    def forward(self, x):
        x = self.pwconv(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        lo = self.dila2(x1)
        hi = self.dila1(x2) - lo
        return self.R(hi), self.R(lo)


class FreqMixer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(FreqMixer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.fd = FreqDec(self.dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        hi ,lo = self.fd(x)
        low_q = rearrange(lo, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(hi, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(hi, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        low_q = torch.nn.functional.normalize(low_q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (low_q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)     # Sparse Attention due to ReLU's property
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class  Sparse_Highpass_Filter(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Sparse_Highpass_Filter, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()
        ####
        self.Maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),nn.GELU())
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v= qkv.chunk(3, dim=1)
        hi_q = self.Maxpool(q)
        hi_q = rearrange(hi_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        hi_q = torch.nn.functional.normalize(hi_q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (hi_q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)     # Sparse Attention due to ReLU's property
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # high = self.Maxpool(g)
        # import pdb; pdb.set_trace()
        # end = out * high
        out = self.project_out(out)
        return out

class Sparse_Lowpass_Filter(nn.Module):  ##
    def __init__(self, dim, num_heads, bias):
        super(Sparse_Lowpass_Filter,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), nn.GELU())

    def upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # B, C, H, W
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        low_q = self.upsample(self.pool(q), size=q.shape[2:])
        low_q = rearrange(low_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        low_q = torch.nn.functional.normalize(low_q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (low_q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)     # Sparse Attention due to ReLU's property
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, dim, ffn_expansion_factor, act_layer=nn.GELU):
        super(Mlp,self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, dim, 1)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x
#########################################

class AFreqMixer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(AFreqMixer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.FreqMixer = FreqMixer(dim, num_heads, bias=False)
        self.ffn = Mlp(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = self.FreqMixer(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class HighpassMixer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(HighpassMixer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.Sparse_Highpass_Filter = Sparse_Highpass_Filter(dim, num_heads, bias=False)
        self.ffn = Mlp(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = self.Sparse_Highpass_Filter(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class LowpassMixer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(LowpassMixer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.Sparse_Lowpass_Filter = Sparse_Lowpass_Filter(dim, num_heads, bias=False)
        self.ffn = Mlp(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = self.Sparse_Lowpass_Filter(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

##########################
class Freq_Pyramid(nn.Module):
    def __init__(self, 
        inp_channels, 
        dim,
        num_heads):
        super(Freq_Pyramid, self).__init__()
        self.PatchEmbed = OverlapPatchEmbed(inp_channels , dim[0])
        self.freqdec0 = FreqDec(dim[0])
        self.freqdec1 = FreqDec(dim[1])
        self.freqdec2 = FreqDec(dim[2])
        self.downsample_01 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.downsample_12 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.increase_1 = nn.Conv2d(int(dim[0]), int(dim[1]), kernel_size=1)
        self.increase_2 = nn.Conv2d(int(dim[1]), int(dim[2]), kernel_size=1)
        self.highfilter_0 = nn.Sequential(*[HighpassMixer(dim=dim[0],num_heads=num_heads[0], ffn_expansion_factor=2) for i in range(2)])
        self.highfilter_1 = nn.Sequential(*[HighpassMixer(dim=dim[1],num_heads=num_heads[1], ffn_expansion_factor=2) for i in range(4)])
        self.highfilter_2 = nn.Sequential(*[HighpassMixer(dim=dim[2],num_heads=num_heads[2], ffn_expansion_factor=2) for i in range(4)])

    def forward(self, x):
        x0 = self.PatchEmbed(x)
        hi0, lo0 = self.freqdec0(x0)
        x1 = self.increase_1(self.downsample_01(lo0))
        hi1, lo1 = self.freqdec1(x1)
        x2 = self.increase_2(self.downsample_12(lo1))
        hi2, lo2 = self.freqdec2(x2)

        high0 = self.highfilter_0(hi0) #torch.Size([1, 24, 192, 192])
        high1 = self.highfilter_1(hi1) #torch.Size([1, 48, 96, 96])
        high2 = self.highfilter_2(hi2)

        return high0, high1, high2, lo2

class Manipulator(nn.Module):
    def __init__(self,dim):
        super(Manipulator, self).__init__()
        self.dim = dim[2]
        self.lowpass = nn.Sequential(*[LowpassMixer(dim=self.dim, num_heads=8, ffn_expansion_factor=2) for i in range(4)])
        self.nonlinear1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size = 1,padding=0, bias=False),
            nn.GELU())
        self.nonlinear2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size = 1,padding=0, bias=False),
            nn.GELU())
        
    def forward(self, x2_a, x2_b, amp):
        dif2 = x2_b - x2_a
        Th_dif2 = self.lowpass(dif2)
        mag2 = x2_b + self.nonlinear2(self.nonlinear1(Th_dif2) * (amp-1))
        return  mag2

class Pyramid_recons(nn.Module):
    def __init__(self, 
        out_channels, 
        num_heads,
        dim):
        super(Pyramid_recons, self).__init__()
        self.fuse_0 = nn.Sequential(*[AFreqMixer(dim=dim[0],num_heads=num_heads[0],  ffn_expansion_factor=2) for i in range(6)])
        self.fuse_1 = nn.Sequential(*[AFreqMixer(dim=dim[1],num_heads=num_heads[1],  ffn_expansion_factor=2) for i in range(4)])
        self.fuse_2 = nn.Sequential(*[AFreqMixer(dim=dim[2],num_heads=num_heads[2],  ffn_expansion_factor=2) for i in range(4)])
        self.up_2 = Upsample(int(dim[2]))
        self.up_1 = Upsample(int(dim[1]))
        self.up_0 = Upsample(int(dim[0]))
        self.reduce_2 = nn.Conv2d(int(dim[2]*2), int(dim[2]), kernel_size=1)
        self.reduce_1 = nn.Conv2d(int(dim[1]*2), int(dim[1]), kernel_size=1)
        self.reduce_0 = nn.Conv2d(int(dim[0]*2), int(dim[0]), kernel_size=1)

        self.output = nn.Conv2d(int(dim[0]/2), out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, high0,high1,high2,mag2):
        fuse2 = self.reduce_2(torch.cat([mag2, high2],1))
        att_fuse2 = self.fuse_2(fuse2)
        up_fuse2 = self.up_2(att_fuse2)
        fuse1 = self.reduce_1(torch.cat([up_fuse2, high1],1))
        att_fuse1 = self.fuse_1(fuse1)
        up_fuse1 = self.up_1(att_fuse1)
        fuse0 = self.reduce_0(torch.cat([up_fuse1, high0],1))
        att_fuse0 = self.fuse_0(fuse0)
        up_fuse0 = self.up_0(att_fuse0)
        end = self.output(up_fuse0) 
        return end


class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.freq_pyramid = Freq_Pyramid(inp_channels=3, dim = [24, 48, 96], num_heads = [4,4,8])
        self.manipulator = Manipulator(dim= [24, 48, 96])
        self.pyramid_recons = Pyramid_recons(out_channels=3, dim = [24, 48, 96], num_heads = [4,4,8])

    def forward(self, x_a, x_b, amp, mode): # v: texture, m: shape
        if mode == 'train':
            high0_a, high1_a, high2_a, lo2_a = self.freq_pyramid(x_a)  # torch.Size([1, 48, 192, 192])
            high0_b, high1_b, high2_b, lo2_b = self.freq_pyramid(x_b)
            # high0_c, high1_c, high2_c, lo2_c = self.freq_pyramid(x_c)
            mag2 = self.manipulator(lo2_a, lo2_b, amp)
            y_hat = self.pyramid_recons(high0_b, high1_b, high2_b, mag2)
            return y_hat
        elif mode == 'evaluate':
            high0_a, high1_a, high2_a, lo2_a = self.freq_pyramid(x_a)  # torch.Size([1, 48, 192, 192])
            high0_b, high1_b, high2_b, lo2_b = self.freq_pyramid(x_b)
            ###############
            m_enc = self.manipulator(lo2_a, lo2_b, amp)
            y_hat = self.pyramid_recons(high0_b, high1_b, high2_b, m_enc)
            return y_hat
