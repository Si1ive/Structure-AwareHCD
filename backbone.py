# -*- coding:utf-8 -*-
# Author:Ding
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba.spatialmamba import SpatialMambaBlock


# band-wise fusion
class BandWiseFusion(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        out_planes = in_planes // 2
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1,
                               groups=out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.gelu1 = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        x = torch.zeros([x1.shape[0], 2 * x1.shape[1], x1.shape[2], x1.shape[3]]).to(x1.device)
        x[:, 0::2, ...] = x1
        x[:, 1::2, ...] = x2
        # TODO 是否需要在融合之前先进行依次逐通道的卷积
        out = self.dropout(self.gelu1(self.bn1(self.conv1(x))))

        return out

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(stride,stride),padding=0, bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),padding=1, bias=False)

class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CNNBlock, self).__init__()
        self.conv11 = conv1x1(in_planes, out_planes)
        self.bn11 = nn.BatchNorm2d(out_planes)
        self.relu11 = nn.ReLU()
        self.conv12 = conv3x3(out_planes, out_planes//2)

        self.conv21 = conv3x3(in_planes, out_planes)
        self.bn21 = nn.BatchNorm2d(out_planes)
        self.relu21 = nn.ReLU()
        self.conv22 = conv3x3(out_planes, out_planes//2)

    def forward(self,x):
        f1 = self.relu11(self.bn11(self.conv11(x)))
        f1 = self.conv12(f1)
        f2 = self.relu21(self.bn21(self.conv21(x)))
        f2 = self.conv22(f2)
        x = torch.cat([f1, f2],dim=1)
        return x  # [B, out_planes, H,W]


class TemporalSpectrumBlock(nn.Module):
    def __init__(self, dim):
        super(TemporalSpectrumBlock, self).__init__()
        self.temporalspectrummamba = SpatialMambaBlock(hidden_dim=dim, drop_path=0., attn_drop_rate=0.)

    def forward(self, x, y):
        diff = torch.abs(x - y)
        x = self.temporalspectrummamba(diff)
        return x


# 定义骨干网络
class SAHCD(nn.Module):
    def __init__(self, loss_strategy, dims):
        super(SAHCD, self).__init__()
        self.loss_strategy = loss_strategy
        #空间
        self.conv1 = CNNBlock(dims[0], dims[1])
        self.spatial1 = SpatialMambaBlock(hidden_dim=dims[1],drop_path=0.,attn_drop_rate=0.)
        self.conv2 = CNNBlock(2 * dims[1], dims[2])
        self.spatial2 = SpatialMambaBlock(hidden_dim=dims[2],drop_path=0.,attn_drop_rate=0.)
        self.conv3 = CNNBlock(2 * dims[2], dims[3])

        #时相-光谱
        self.temporalspectrumblock1 = TemporalSpectrumBlock(dims[1])
        self.temporalspectrumblock2 = TemporalSpectrumBlock(dims[2])
        self.temporalspectrumblock3 = TemporalSpectrumBlock(dims[3])

        #分类
        in_dim = dims[1] + dims[2] + dims[3]
        self.CD = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1), bias=False),
                                nn.ReLU(inplace=True),  # third layer
                                nn.Dropout(0.1),
                                nn.Conv2d(in_dim, in_dim // 2, kernel_size=(1, 1), bias=False),
                                nn.ReLU(inplace=True),  # third layer
                                nn.Dropout(0.1),
                                nn.Conv2d(in_dim // 2, 2, kernel_size=(1, 1), bias=True))

    def _make_spAtt_layer(self, block, in_dim, out_dim, hidn_dim):
            return nn.Sequential(block(in_dim, out_dim, hidn_dim))

    #空间
    def getFeature(self, x):
        #第一层
        f1 = self.conv1(x)  # B, C, H,W
        #第二层
        f1_ = self.spatial1(f1)
        f1_ = torch.cat([f1, f1_], dim=1)
        f2 = self.conv2(f1_)
        #第三层
        f2_ = self.spatial2(f2)
        f2_ = torch.cat([f2, f2_], dim=1)
        f3 = self.conv3(f2_)
        return f1, f2, f3

    #时相-光谱
    def getDifference(self, f10, f20, f11, f21, f12, f22):
        f1 = self.temporalspectrumblock1(f10, f20)
        f2 = self.temporalspectrumblock2(f11, f21)
        f3 = self.temporalspectrumblock3(f12, f22)
        f = torch.cat([f1, f2, f3], dim=1)  # B,C,H,W
        out = self.CD(f).squeeze()   # 2,H,W
        return out
        # to visiualize the feature map

    def cross_entropy(self, loss_fuc1, result, label, idx):
        num, H, B = result.shape
        result = result.reshape([num, -1])  # [2, H*W]
        result_dx = result[:, idx]   # [2, N]
        result_dx = result_dx.transpose(1, 0)   # [N, 2]
        l_ce = loss_fuc1(result_dx, label.squeeze())
        return l_ce

    def forward(self, t1, t2, idx, label, loss_fuc1):
        f10, f11, f12 = self.getFeature(t1)
        f20, f21, f22 = self.getFeature(t2)
        if self.loss_strategy == 'single':
            output = self.getDifference(f10, f20, f11, f21, f12, f22)
            if self.training:
                l_ce = self.cross_entropy(loss_fuc1, output, label, idx)
                return l_ce
            return output
        elif self.loss_strategy == 'double':
            output1 = self.getDifference(f10, f20, f11, f21, f12, f22)
            output2 = self.getDifference(f20, f10, f21, f11, f22, f12)
            if self.training:
                l_ce1 = self.cross_entropy(loss_fuc1, output1, label, idx)
                l_ce2 = self.cross_entropy(loss_fuc1, output2, label, idx)
                return (l_ce1 + l_ce2) / 2
            return output1, output2




