import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import numpy.random as random



#DF-GAN
class _NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(_NetG, self).__init__()
        self.ngf = ngf# 生成器feature map数(计算通道数量的一倍增量值)
        # input noise (batch_size, 100)
        # fc层输出的结果进入第一个size为4的df层（（ngf*8）*（4*4））------ngf*2**n为不同df层的通道扩充
        self.fc = nn.Linear(nz, ngf*8*4*4)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        # [i，o]：[8,8][8,8][8,8][8,4][4,2][2,1]
        # channel_nums：[nf * i，nf * o]
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            #最后一个是32 * 1的out_ch，通过卷积提取特征生成3个通道的rgb图像
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
            )
    
    def forward(self, noise, c): # x=noise, c=ent_emb
        # concat noise and sentence
        out = self.fc(noise)
        # noise.size(0)为batch_size
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        for GBlock in self.GBlocks:
            out = GBlock(x=out, y=cond, y_noZ=None)


            #最后输出batch_size *（32*1）* 256 * 256
        # convert to RGB image
        out = self.to_rgb(out)
        return out














#并联
# class NetG(nn.Module):
#     def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
#         super(NetG, self).__init__()
#         self.ngf = ngf  # 生成器feature map数(计算通道数量的一倍增量值)
#         # input noise (batch_size, 100)
#         # fc层输出的结果进入第一个size为4的df层（（ngf*8）*（4*4））------ngf*2**n为不同df层的通道扩充
#         self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
#         # build GBlocks
#         self.GBlocks = nn.ModuleList([])
#         # local_GBlocks
#         self.local_GBlocks = nn.ModuleList([])
#         self.mix_CNNs = nn.Sequential(
#                     nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1),
#                     nn.LeakyReLU(0.2, inplace=True),
#                     nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1))
#
#         # [i，o]：[8,8][8,8][8,8][8,4][4,2][2,1]
#         # channel_nums：[nf * i，nf * o]
#         in_out_pairs = get_G_in_out_chs(ngf, imsize)
#         for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
#             self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
#             if out_ch==ngf * 8:
#                 self.local_GBlocks.append(G_Block_local(cond_dim, in_ch, out_ch, upsample=True))
#
#         # to RGB image
#         self.to_rgb = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             # 最后一个是32 * 1的out_ch，通过卷积提取特征生成3个通道的rgb图像
#             nn.Conv2d(out_ch, ch_size, 3, 1, 1),
#             nn.Tanh(),
#         )
#
#     
#     def forward(self, noise, c, w, mask):  # x=noise, c=ent_emb
#         # concat noise and sentence
#         out = self.fc(noise)
#         # noise.size(0)为batch_size
#         out = out.view(noise.size(0), 8 * self.ngf, 4, 4)
#         cond = torch.cat((noise, c), dim=1)
#         # fuse text and visual features
#         local_out = out
#         weight = []
#         for GBlock_local in self.local_GBlocks:
#             local_out, local_weight = GBlock_local(x=local_out, y=w, mask=mask)
#             # 总共(2*2)*3=12个words_affineBlock，len(weight)=3，
#             # weight[i]为第i个local_GBlock（总共3个），weight[i][j]为第i个local_GBlock的第j个DF模块（总共2个）,weight[i][j][k]为第i个local_GBlock的第j个DF模块的第k个仿射模块（总共2个）
#             weight.append(local_weight)
#         for GBlock in self.GBlocks:
#             out = GBlock(x=out, y=cond, y_noZ=c)
#             if out.shape[-1]==local_out.shape[-1]:
#                 out = torch.cat((out, local_out,), dim=1)
#                 out = self.mix_CNNs(out)
#
#             # 最后输出batch_size *（32*1）* 256 * 256
#         # convert to RGB image
#         out = self.to_rgb(out)
#         return out, weight

#串联
class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetG, self).__init__()
        self.ngf = ngf  # 生成器feature map数(计算通道数量的一倍增量值)
        # input noise (batch_size, 100)
        # fc层输出的结果进入第一个size为4的df层（（ngf*8）*（4*4））------ngf*2**n为不同df层的通道扩充
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])

        # [i，o]：[8,8][8,8][8,8][8,4][4,2][2,1]
        # channel_nums：[nf * i，nf * o]
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if out_ch==ngf * 8:
                self.GBlocks.append(G_Block_mix(cond_dim, in_ch, out_ch, upsample=True))
            else:
                self.GBlocks.append(G_Block(cond_dim + nz, in_ch, out_ch, upsample=True))

        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # 最后一个是32 * 1的out_ch，通过卷积提取特征生成3个通道的rgb图像
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
        )

    
    def forward(self, noise, c, w, mask):  # x=noise, c=ent_emb
        # concat noise and sentence
        out = self.fc(noise)
        # noise.size(0)为batch_size
        out = out.view(noise.size(0), 8 * self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        weight = []
        for idx, GBlock in enumerate(self.GBlocks):
            if idx<=2:
                out, local_weight = GBlock(x=out, y=cond, w=w, mask=mask)
                weight.append(local_weight)
            else:
                out = GBlock(x=out, y=cond, y_noZ=c)

            # 最后输出batch_size *（32*1）* 256 * 256
        # convert to RGB image
        out = self.to_rgb(out)
        return out, weight


# global_switchTrain = True
# #both_all串联
# class NetG(nn.Module):
#     def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
#         super(NetG, self).__init__()
#         self.ngf = ngf  # 生成器feature map数(计算通道数量的一倍增量值)
#         # input noise (batch_size, 100)
#         # fc层输出的结果进入第一个size为4的df层（（ngf*8）*（4*4））------ngf*2**n为不同df层的通道扩充
#         self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
#         # build GBlocks
#         self.GBlocks = nn.ModuleList([])
#
#         # [i，o]：[8,8][8,8][8,8][8,4][4,2][2,1]
#         # channel_nums：[nf * i，nf * o]
#         in_out_pairs = get_G_in_out_chs(ngf, imsize)
#         for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
#             self.GBlocks.append(G_Block_mix(cond_dim, in_ch, out_ch, upsample=True))
#
#
#         # to RGB image
#         self.to_rgb = nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=True),
#             # 最后一个是32 * 1的out_ch，通过卷积提取特征生成3个通道的rgb图像
#             nn.Conv2d(out_ch, ch_size, 3, 1, 1),
#             nn.Tanh(),
#         )
#
#     
#     def forward(self, noise, c, w, mask):  # x=noise, c=ent_emb
#         # global global_switchTrain
#         # if switchTrain:
#         #     global_switchTrain=True
#         # else:
#         #     global_switchTrain=False
#         # concat noise and sentence
#         out = self.fc(noise)
#         # noise.size(0)为batch_size
#         out = out.view(noise.size(0), 8 * self.ngf, 4, 4)
#         cond = torch.cat((noise, c), dim=1)
#         # fuse text and visual features
#         weight = []
#         for idx, GBlock in enumerate(self.GBlocks):
#             # print(idx)
#             out, local_weight = GBlock(x=out, y=cond, w=w, mask=mask)
#             weight.append(local_weight)
#
#             # 最后输出batch_size *（32*1）* 256 * 256
#         # convert to RGB image
#         out = self.to_rgb(out)
#         return out, weight




class G_Block_mix(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block_mix, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.in_ch = in_ch
        self.out_ch = out_ch

        #3*3尺度
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.fuse1 = local_DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim+100, out_ch)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)



    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y, w, mask):

        h, weight0 = self.fuse1(h, w, mask)
        h = self.c1(h)
        # h, weight1 = self.fuse2(h, y, mask)
        h = self.fuse2(h, y)
        h = self.c2(h)
        # weight = []
        # weight.append(weight0)
        # weight.append(weight1)

        return h, weight0

    def forward(self, x, y, w, mask):
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)
        res = self.residual(x, y, w, mask)
        return self.shortcut(x) + res[0], res[1]

class G_Block_local(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block_local, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.in_ch = in_ch
        self.out_ch = out_ch

        #3*3尺度
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.fuse1 = local_DFBLK(cond_dim, in_ch)
        self.fuse2 = local_DFBLK(cond_dim, out_ch)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)



    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y, mask):

        h, weight0 = self.fuse1(h, y, mask)
        h = self.c1(h)
        h, weight1 = self.fuse2(h, y, mask)
        h = self.c2(h)
        weight = []
        weight.append(weight0)
        weight.append(weight1)

        return h, weight

    def forward(self, x, y, mask):
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)
        res = self.residual(x, y, mask)
        return self.shortcut(x) + res[0], res[1]


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        

        #3*3尺度
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)


    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y, y_noZ):

        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)

        return h

    def forward(self, x, y, y_noZ):
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)

        return self.shortcut(x) + self.residual(x, y, y_noZ)


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.cond_dim = cond_dim

        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):

        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)

        return h

class local_DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(local_DFBLK, self).__init__()
        self.cond_dim = cond_dim

        self.affine0 = Affine_word(cond_dim, in_ch)
        self.affine1 = Affine_word(cond_dim, in_ch)

    def forward(self, x, y=None, mask=None):

        h, weight0 = self.affine0(x, y, mask)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h, weight1 = self.affine1(h, y, mask)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        weight = []
        weight.append(weight0)
        weight.append(weight1)

        return h, weight


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, cAttn=None, sAttn=None, attn=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        

        return weight * x + bias


class Affine_word(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine_word, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_words_weight = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, int(cond_dim / 2))),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(int(cond_dim / 2), 1)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)
        nn.init.zeros_(self.fc_words_weight.linear2.weight.data)
        nn.init.zeros_(self.fc_words_weight.linear2.bias.data)

    def forward(self, x, y=None, mask=None):
        cond = y
        cond = torch.transpose(cond, 1, 2)
        weight = self.fc_gamma(cond)
        bias = self.fc_beta(cond)
        #给每个单词增加权重
        attn_weight = self.fc_words_weight(cond).clone()

        for idx, _ in enumerate(attn_weight):
            attn_weight[idx, mask[idx]:, :] = -float('inf')
        attn_weight = nn.Softmax(dim=1)(attn_weight)
        re_weight = attn_weight

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        if x.shape[-1]<64:
            size = [cond.shape[0], cond.shape[1], x.shape[-3], x.shape[-2], x.shape[-1]]

            weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
            bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
            x = x.unsqueeze(1).expand(size)
            attn_weight = attn_weight.unsqueeze(-1).unsqueeze(-1).expand(size)

            x = attn_weight * (weight * x + bias)
            x = torch.sum(x, dim=1)
        else:
            size = [cond.shape[0], 1, x.shape[-3], x.shape[-2], x.shape[-1]]
            weight = weight * attn_weight
            bias = bias * attn_weight
            weight = weight.unsqueeze(-1).unsqueeze(-1)
            bias = bias.unsqueeze(-1).unsqueeze(-1)
            x = x.unsqueeze(1)
            out = None

            for i in range(cond.shape[1]):
                iweight = weight[:, i, :, :, :].unsqueeze(1).expand(size)
                ibias = bias[:, i, :, :, :].unsqueeze(1).expand(size)
                if i==0:
                    out = iweight * x + ibias
                else:
                    # print(i)
                    out = torch.concat((out, iweight * x + ibias), dim=1)
                    out = torch.sum(out, dim=1).unsqueeze(1)

            x = out.squeeze(1)



        # padding = torch.zeros([x.shape[-3], x.shape[-2], x.shape[-1]], dtype=torch.float)
        # for idx, _ in enumerate(x):
        #     x[idx, mask[idx]:, :, :, :] = padding
        # x = torch.sum(x, dim=1) / cond.shape[1]

        # x = torch.sum(x, dim=1)

        # #归一化
        # for idx, _ in enumerate(x):
        #     x[idx, :, :, :] = x[idx, :, :, :] / mask[idx]

        # print(re_weight)
        # print(torch.sum(re_weight, dim=1))
        return x, re_weight



# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        #size不变，扩充通道数量
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        # [1,2][2,4][4,8][8,8][8,8][8,8]
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))


    def forward(self,x):
        #先将3通道转换为第一个DownBlock所需的通道数ndf * 1（32*1）
        out = self.conv_img(x)
        # localF = None
        for DBlock in self.DBlocks:
            out = DBlock(out)
            # # 最后输出batch_size *（32*8）* 4 * 4
            # if out.shape[3]==16:
            #     localF = out
            #     #取256 * 16 * 16 作为局部特征

        globalF = out

        return globalF#, localF
#原始的NetC
class _NetC(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(_NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf*8+cond_dim, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
        )


    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)

        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out




class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            #第一个卷积使得size减半，且通道增加
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        #torch.zeros(1) --> tensor([0.])
        #nn.Parameter()将这个设置为可训练的参数
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):

        #残差块变化了通道数量和size
        res = self.conv_r(x)
        #为了融合残差块，x需要变换通道和size
        if self.learned_shortcut:
            #size不变，通道增加
            x = self.conv_s(x)
        if self.downsample:
            #按照2倍缩小进行下采样
            x = F.avg_pool2d(x, 2)

        return x + self.gamma*res








def get_G_in_out_chs(nf, imsize):
    #7层
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    #[8,8][8,8][8,8][8,4][4,2][2,1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    #[1,2][2,4][4,8][8,8][8,8][8,8]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs
