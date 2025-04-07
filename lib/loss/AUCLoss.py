import torch
import torch.nn as nn
import torch.nn.functional as F

class AUCLoss(nn.Module):
    def __init__(self, gamma=1.0):
        """
        AUC 近似损失函数。
        :param gamma: 调节参数，控制损失函数的平滑程度。
        """
        super(AUCLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        计算 AUC 近似损失。
        :param logits: 模型的预测值，形状为 [batch_size]。
        :param labels: 真实标签，形状为 [batch_size]，值为 0 或 1。
        :return: AUC 近似损失值。
        """
        # 将正样本和负样本分开
        positive_logits = logits[labels == 1]
        negative_logits = logits[labels == 0]

        # 计算所有正负样本对的预测值差异
        # 使用广播机制，计算每个正样本与每个负样本的预测值差异
        differences = positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)  # [num_pos, num_neg]

        # 使用 Softplus 函数近似阶跃函数，增加可微性
        # Softplus(x) = log(1 + exp(x))，是一个平滑的近似 ReLU 函数
        losses = F.softplus(-self.gamma * differences)  # [num_pos, num_neg]

        # 计算平均损失
        auc_loss = losses.mean()

        return auc_loss