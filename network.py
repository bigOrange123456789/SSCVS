# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from base_model import resnet50
from base_model import UNet,UNet_IBN, UNet_DA, UNet_contrast, UNet_contrastbase, Iternet, Iternet_constrast

from config import config

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(Network, self).__init__()
        self.branch1 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.branch2 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)

class Network_UNet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(Network_UNet, self).__init__()
        self.branch1 = SingleUNet(n_channels,num_classes)
        self.branch2 = SingleUNet(n_channels,num_classes)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)

class Network_UNet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(Network_UNet, self).__init__()
        self.branch1 = SingleUNet(n_channels,num_classes)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


class Network_Iternet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(Network_Iternet, self).__init__()
        self.branch1 = SingleIternet(n_channels,num_classes)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        return self.branch1(data)


class SingleUNet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(SingleUNet, self).__init__()
        self.backbone = UNet(n_channels=n_channels, n_classes=num_classes)
            # (pretrained_model, norm_layer=norm_layer,
            #                       bn_eps=config.bn_eps,
            #                       bn_momentum=config.bn_momentum,
            #                       deep_stem=True, stem_width=64)
        # self.dilate = 2
        # for m in self.backbone.layer4.children():
        #     m.apply(partial(self._nostride_dilate, dilate=self.dilate))
        #     self.dilate *= 2
        # self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.backbone)
        # self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        # self.business_layer.append(self.classifier)

    def forward(self, data):
        pred = self.backbone(data)
        # v3plus_feature = self.head(blocks)      # (b, c, h, w)
        # b, c, h, w = v3plus_feature.shape
        #
        # pred = self.classifier(v3plus_feature)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        # if self.training:
        #     return pred
        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class SingleIternet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(SingleIternet, self).__init__()
        self.backbone = Iternet(n_channels=n_channels, n_classes=num_classes)
        self.business_layer = []
        self.business_layer.append(self.backbone)

    def forward(self, data):
        pred = self.backbone(data)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class Single_contrast_UNet(nn.Module): # Single_contrast_UNet 单对比度UNet
    # 在PyTorch中，模型是通过继承torch.nn.Module类来定义的，这使得模型能够利用PyTorch提供的各种神经网络构建块和功能。
    def __init__(self, n_channels,num_classes): # 4,1
        super(Single_contrast_UNet, self).__init__()
        self.backbone = UNet_contrast(n_channels=n_channels, n_classes=num_classes) # 通道=4,分类=1
        self.business_layer = []
        self.business_layer.append(self.backbone)
        # self.learnable_scalar = nn.Parameter(torch.tensor(1.0))  # 用于对比学习的标量
        # self.learnable_scalar = self.backbone.learnable_scalar
        if config['marginInfoNCE']:
            self.learnable_scalar = nn.Parameter(torch.tensor(2.0))  # 用于对比学习的标量

    def forward(self, data,mask=None,trained=True, fake=True): # fake=T/F -> masks真标签/预测标签
        '''
            mask=None：
                数据标签。
            trained=True：
                这个参数可能用于控制模型是否处于训练模式。
                在某些情况下，模型在训练时的行为（如dropout和批量归一化）与在评估或推理时的行为不同。
            fake=True：
                fake的含义是区分 真实血管/合成血管
                fake=T/F -> masks真标签/预测标签
                因为fake只影响对比学习，所以只在训练时有用、在评估时没用。
        '''
        pred, sample_set, flag, feature = self.backbone(data,mask,trained, fake)
        # pred：      类0-1标签的预测结果
        # sample_set：用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        # no use? pred is the same size as data (h,w) 没用？pred的大小与数据（h，w）相同
        '''
        对pred进行上采样或下采样，即改变数据的空间尺寸。
            'bilinear'双线性插值，这是一种在二维空间中常用的插值方法，可以生成相对平滑的图像。
            align_corners=True：输入和输出张量的角点（即左上角和右下角）会被对齐，这通常用于保持图像边角的一致性。
        '''

        return pred, sample_set, flag, feature

    # @staticmethod
    def _nostride_dilate(self, m, dilate): # 无步长膨胀
        '''
            以下划线（_）开头的标识符通常表示这是一个受保护的成员或私有成员，意味着它主要用于类内部使用，而不应该被类的外部直接访问。
                m 可能代表某种数据结构或参数矩阵，
                dilate可能是一个用于膨胀操作的参数，比如膨胀的因子或模式。
        '''
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class Single_contrast_Iternet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(Single_contrast_Iternet, self).__init__()
        self.backbone = Iternet_constrast(n_channels=n_channels, n_classes=num_classes)
            # (pretrained_model, norm_layer=norm_layer,
            #                       bn_eps=config.bn_eps,
            #                       bn_momentum=config.bn_momentum,
            #                       deep_stem=True, stem_width=64)
        # self.dilate = 2
        # for m in self.backbone.layer4.children():
        #     m.apply(partial(self._nostride_dilate, dilate=self.dilate))
        #     self.dilate *= 2
        # self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.backbone)
        # self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        # self.business_layer.append(self.classifier)

    def forward(self, data,mask=None,trained=True, fake=True):
        pred, sample_set, flag = self.backbone(data,mask,trained, fake)
        # v3plus_feature = self.head(blocks)      # (b, c, h, w)
        # b, c, h, w = v3plus_feature.shape
        #
        # pred = self.classifier(v3plus_feature)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        # if self.training:
        #     return pred
        return pred, sample_set, flag

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)



class Single_contrast_UNetbase(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(Single_contrast_UNetbase, self).__init__()
        self.backbone = UNet_contrastbase(n_channels=n_channels, n_classes=num_classes)
            # (pretrained_model, norm_layer=norm_layer,
            #                       bn_eps=config.bn_eps,
            #                       bn_momentum=config.bn_momentum,
            #                       deep_stem=True, stem_width=64)
        # self.dilate = 2
        # for m in self.backbone.layer4.children():
        #     m.apply(partial(self._nostride_dilate, dilate=self.dilate))
        #     self.dilate *= 2
        # self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.backbone)
        # self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        # self.business_layer.append(self.classifier)

    def forward(self, data,mask=None,trained=True, fake=True):
        pred, sample_set, flag, d2 = self.backbone(data,mask,trained, fake)
        # v3plus_feature = self.head(blocks)      # (b, c, h, w)
        # b, c, h, w = v3plus_feature.shape
        #
        # pred = self.classifier(v3plus_feature)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        # if self.training:
        #     return pred
        return pred, sample_set, flag, d2

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class Single_IBNUNet(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(Single_IBNUNet, self).__init__()
        self.backbone = UNet_IBN(n_channels=n_channels, n_classes=num_classes)
        self.business_layer = []
        self.business_layer.append(self.backbone)

    def forward(self, data):
        pred = self.backbone(data)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class SingleUNet_featureDA(nn.Module):
    def __init__(self, n_channels,num_classes):
        super(SingleUNet_featureDA, self).__init__()
        self.backbone = UNet_DA(n_channels=n_channels, n_classes=num_classes)
        self.business_layer = []
        self.business_layer.append(self.backbone)

    def forward(self, data):
        pred, center_feature = self.backbone(data)
        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        return pred, center_feature

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class SingleNetwork(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(SingleNetwork, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)

    def forward(self, data):
        blocks = self.backbone(data)
        v3plus_feature = self.head(blocks)      # (b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            return v3plus_feature, pred
        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        return f


if __name__ == '__main__':
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
