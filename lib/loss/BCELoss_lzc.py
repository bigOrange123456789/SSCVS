import torch
import torch.nn as nn
def asymmetric_loss(x, y, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
# def asymmetric_loss(x, y, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
    gamma_neg=0#config.gamma_neg
    gamma_pos=0#config.gamma_pos
    """"
    用于计算不平衡损失
    源自谢驰师哥的代码：https://github.com/charles-xie/CQL
    Parameters
    ----------
    x: input logits     [4, 1, 256, 256]
    tensor([[[[0.7753, 0.7117, 0.5816,  ...,
    y: targets (multi-label binarized vector)   [4, 1, 256, 256]
    tensor([[[[0., 0., 0.,  ...,
    """
    # criterion_bce(x=pred_sup_l, y=gts) # 根据预测结果x和标签y计算CE损失
    pos_inds = y.eq(1).float() #转换为布尔型,再转换为浮点型
    num_pos = pos_inds.float().sum() #标签中血管所占像素的总数
    '''
    x.shape=[4, 1, 256, 256]
    y.shape=[4, 1, 256, 256]
    num_pos=tensor(19957.)
    '''

    # Calculating Probabilities
    # x_sigmoid = torch.sigmoid(x)
    x_sigmoid = x  # x.shape=[4, 1, 256, 256]
    xs_pos = x_sigmoid #血管概率
    xs_neg = 1 - x_sigmoid #背景概率
    # print(xs_pos.shape, xs_neg.shape,xs_neg)

    # Asymmetric Clipping 不对称剪裁(这个操作的作用是什么？)
    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1) # 将所有背景概率都增大clip(5%)

    # Basic CE calculation
    los_pos = y * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - y) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg # loss.shape = [4, 1, 256, 256]

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        if disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False) # 接下来不去计算梯度
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True) # 接下来恢复计算梯度
        loss *= one_sided_w
    # loss.shape=[4, 1, 256, 256] <torch.Tensor>

    if num_pos == 0:
        return -loss.sum()
    else:
        return -loss.sum() / num_pos

def bce_loss_lzc(x, y, eps=1e-8):

    # Calculating Probabilities
    # x_sigmoid = torch.sigmoid(x)
    x_sigmoid = x  # x.shape=[4, 1, 256, 256]
    xs_pos = x_sigmoid #血管概率
    xs_neg = 1 - x_sigmoid #背景概率
    # print(xs_pos.shape, xs_neg.shape,xs_neg)

    # Basic CE calculation
    los_pos = y * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - y) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg # loss.shape = [4, 1, 256, 256]

    return -loss.mean()
class BCELoss_lzc(nn.Module):
    def __init__(
        self,
        weight=None,
        eps=1e-8,
        gamma_neg=0,
        gamma_pos=0,
        disable_torch_grad_focal_loss=True
    ):
        super(BCELoss_lzc, self).__init__()
        self.weight=weight
        self.eps=eps
        self.gamma_neg=gamma_neg #用于Focal Loss
        self.gamma_pos=gamma_pos #用于Focal Loss
        self.disable_torch_grad_focal_loss=disable_torch_grad_focal_loss #用于Focal Loss

    def forward(self, x , y ) :
        xs_pos = x #血管概率
        xs_neg = 1 - x #背景概率

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg # loss.shape = [4, 1, 256, 256]


        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False) # 接下来不去计算梯度
            pt = xs_neg * y  + xs_pos * (1 - y)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow( pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True) # 接下来恢复计算梯度
            loss *= one_sided_w

        if not self.weight is None:#原本BCE自带的加权功能
            loss=loss*self.weight

        return -loss.mean()
