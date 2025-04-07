import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from lib.network.UNet_contrast import UNet_contrast

class ModelSegment(nn.Module): # Single_contrast_UNet 单对比度UNet
    # 在PyTorch中，模型是通过继承torch.nn.Module类来定义的，这使得模型能够利用PyTorch提供的各种神经网络构建块和功能。
    def __init__(self, n_channels,num_classes): # 4,1
        super(ModelSegment, self).__init__()
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
        result = self.backbone(data,mask,trained, fake)
        # pred：      类0-1标签的预测结果
        # sample_set：用于对比学习的采样结果(正负像素的数量、特征、均值，正负难易像素的数量、特征)
        b, c, h, w = data.shape
        result["pred"] = F.interpolate(result["pred"], size=(h, w), mode='bilinear', align_corners=True)
        # no use? pred is the same size as data (h,w) 没用？pred的大小与数据（h，w）相同
        '''
        对pred进行上采样或下采样，即改变数据的空间尺寸。
            'bilinear'双线性插值，这是一种在二维空间中常用的插值方法，可以生成相对平滑的图像。
            align_corners=True：输入和输出张量的角点（即左上角和右下角）会被对齐，这通常用于保持图像边角的一致性。
        '''

        return result#pred, sample_set, flag, feature

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

if __name__ == '__main__':
    # model = Network(40, criterion=nn.CrossEntropyLoss(),
    #                 pretrained_model=None,
    #                 norm_layer=nn.BatchNorm2d)
    model = ModelSegment(3,3)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
