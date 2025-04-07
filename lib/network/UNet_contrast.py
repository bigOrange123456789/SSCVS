# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.network.ContrastiveHead_myself import ContrastiveHead_myself
from lib.network.conv_block import conv_block
from config import config
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet_contrast(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes): # 通道=4,分类=1
        super(UNet_contrast, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # reduce size to less than a half
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])   # a self-defined class
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])     # a self-defined class
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid() #(-inf,+inf)->(0,1)
        # self.contrast = ContrastiveHead_torch(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3) #init the contrast head to conv 8;
        if config.contrast["weight"]>0:
            self.contrast = ContrastiveHead_myself(num_convs=1,num_projectfc=2,thred_u=0.1,scale_u=1.0,percent=0.3)
        else:
            self.contrast = None

        # self.learnable_scalar = nn.Parameter(torch.tensor(1000.0))  # 用于对比学习的标量
        self.encoder = nn.Sequential(
            self.Conv1,
            self.Maxpool1, #最大池化层里面应该没有参数
            self.Conv2,
            self.Maxpool2,
            self.Conv3,
            self.Maxpool3,
            self.Conv4,
            self.Maxpool4,
            self.Conv5,
        )
        self.decoder = nn.Sequential(
            self.Up5,
            self.Up_conv5,
            self.Up4,
            self.Up_conv4,
            self.Up3,
            self.Up_conv3,
            self.Up2,
            self.Up_conv2,
            self.Conv
        )#这个主干网络由 “编码器、解码器、对比头”这三部分构成，三部分可优化参数的比例为：40,50,20

    def cat_(self,xe,xd):
        diffY = xe.size()[2] - xd.size()[2]
        diffX = xe.size()[3] - xd.size()[3]
        xd = F.pad(xd, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)) # 对 xd 进行填充，可以保证 xd 的大小与 xe 相同
        xd = torch.cat((xe, xd), dim=1)
        return xd

    def __toEncoder(self,x):
        e1 = self.Conv1(x)  # e1[* 64, 256^] <- x[* 4, 256^]

        e2 = self.Maxpool1(e1)  # 1/2 # e2[* 64, 128^]
        e2 = self.Conv2(e2)  # e2[* 128, 128^]

        e3 = self.Maxpool2(e2)  # 1/4 # e3[* 128, 64^]
        e3 = self.Conv3(e3)  # e3[* 256, 64^]

        e4 = self.Maxpool3(e3)  # 1/8 # e4[* 256, 32^]
        e4 = self.Conv4(e4)  # e4[* 512, 32^]

        e5 = self.Maxpool4(e4)  # 1/16# e5[* 512, 16^]
        e5 = self.Conv5(e5)  # e5[* 1024, 16^]
        return e1,e2,e3,e4,e5

    def __toDecoder(self,code):
        e1, e2, e3, e4, e5 = code

        d5 = self.Up5(e5)  # 1/8  # d5[* 512, 32^]
        d5 = self.cat_(e4, d5)  # d5[* 1024, 32^]
        # d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)  # d5[* 512, 32^]

        d4 = self.Up4(d5)  # 1/4  # d4[* 256, 64^]
        d4 = self.cat_(e3, d4)  # d4[* 512, 64^]
        # d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)  # d4[* 256, 64^]

        d3 = self.Up3(d4)  # 1/2  # d3[* 128, 128^]
        d3 = self.cat_(e2, d3)  # d3[* 256, 128^]
        # d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)  # d3[* 128, 128^]

        d2 = self.Up2(d3)  # 1    # d2[* 64, 256^]
        d2 = self.cat_(e1, d2)  # d2[* 128, 256^]
        # d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # d2[* 64, 256^]
        feature = d2  # 用于一致性正则化
        # d2 = d2 + contrast_tensor0
        # d2 [4, 64, 256, 256] <Tensor>
        score = self.Conv(d2)  # 根据每个像素点的特征转换为打分 # out[* 1, 256^]
        return score, feature
    def __codeDetach(self,code):
        e1, e2, e3, e4, e5 = code
        codeDetach = e1.detach(), e2.detach(), e3.detach(), e4.detach(), e5.detach()
        return codeDetach
    def getPredInterrupt(self,code):
        codeDetach = self.__codeDetach(code)
        score, _ =self.__toDecoder(codeDetach)
        pred = self.active(score)
        return pred

    def forward(self, x, mask, trained,fake):
        '''
        整体架构：
            输入：conv:4->64
            分析：[d2 d3 d4 d5; u5 u4 u3 u2]
            输出：conv:64->1
        down:
            pool:   (c,   n/2 )
            conv:   (c*2, n   )
        up:
            up:     (c/2,   n*2 )？需要仔细研究一下
            cat:    (c*2,   n   )
            conv:   (c/2,   n   )
        '''
        code = self.__toEncoder(x)
        out, feature = self.__toDecoder(code)

        d2 = feature
        d1 = self.active(out)#将结果转换为类0-1标签 # d1[* 1, 256^]

        ### contrastive loss: #对比损失函数
        if self.contrast != None:
            if trained and fake:          # mask是监督值
                contrast_tensor0, sample_sets, flag = self.contrast(d2,mask,trained,fake)   # pos and neg features of synthetic imgs
            elif trained and fake==False: # d1是预测值
                contrast_tensor0, sample_sets, flag = self.contrast(d2, d1, trained, fake)  # pos and neg features of target imgs
            else: #验证
                contrast_tensor0, sample_sets, flag = self.contrast(d2, d1, trained, fake)
        else:
            contrast_tensor0, sample_sets, flag = None,None,None

        result={
            "pred":d1,
            "sample_set":sample_sets,
            "flag":flag,
            "feature":feature,
            "code":code
        }
        return result
        # d1：         类0-1标签的预测结果
        # sample_sets：用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
