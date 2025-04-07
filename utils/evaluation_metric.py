r""" Evaluate mask prediction """
import torch
import numpy as np

# from skimage import morphology
from sklearn.metrics import roc_auc_score
def calAUC(pred,gt):
    '''
    TPR：TP/(TP+FN)。TPR越大意味着TP越大，也就意味着对于测试样本中的所有正例来说，其中大部分都被学习器预测正确。
    FPR：FP/(TN+FP)。FPR越小意味着FP越小、TN越大，也就意味着FPR越小，则对于测试样例中的所有反例来说，其中大部分被学习器预测正确。
    我们希望是TPR大、PFR小。
    原文链接：https://blog.csdn.net/qq_46020653/article/details/119613712
    '''
    #get 1-D
    pred_1D = np.array(pred.cpu()).flatten() # (262144,)<-[1, 1, 512, 512]
    gt_1D = np.array(gt.cpu()).flatten()
    # print("gt_1D",gt_1D.shape)
    # print("pred_1D",pred_1D.shape)
    # print("gt_unqiue",np.unique(gt_1D))
    # print("pred_1D_unqiue",np.unique(pred_1D))
    AUC = roc_auc_score(gt_1D, pred_1D)
    # N = 0
    # P = 0
    # neg_prob = []
    # pos_prob = []
    # for key, value in enumerate(gt_1D):
    #     if(value==1):
    #         P += 1
    #         pos_prob.append(pred_1D[key])
    #     else:
    #         N +=1
    #         neg_prob.append(pred_1D[key])
    # number = 0
    # for pos in pos_prob:
    #     for neg in neg_prob:
    #         if(pos>neg):
    #             number +=1
    #         elif (pos==neg):
    #             number += 0.5
    return AUC

def computeF1(pred, gt):
    """

    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score * 100, precision * 100, recall * 100

def jc(pred, gt):
    """
    Jaccard coefficient
    Computes the Jaccard coefficient between the binary objects in two images.
    计算两幅图像中二值对象之间的Jaccard系数。
    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the object(s) in `reference`.
        It ranges from 0 (no overlap) to 1 (perfect overlap).
        “result”中的对象和“reference”中对象之间的Jaccard系数。它的范围从0（无重叠）到1（完全重叠）。
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    这是一个真实的指标。因此，可以以任何顺序提供二进制图像。
    """
    result = pred.cpu().numpy()
    reference = gt.cpu().numpy()
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference) # 计算交集中的元素数量
    union = np.count_nonzero(result | reference)        # 计算并集中的元素数量

    jc = float(intersection) / float(union) #预测越准确的话，交集约接近并集

    return jc

def compute_allXCAD(pred, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    Acc = (tp+tn)/(tp+fn+tn+fp)
    Sp = tn/(tn+fp+ epsilon)
    jc_score = jc(pred,gt)

    return f1_score * 100, precision * 100, recall * 100, Sp * 100, Acc * 100, jc_score *100

def compute_allRetinal(pred, pred_con, gt):
    """
    input:
        pred,     #预测概率
        pred_con, #预测标签
        gt        #手工标签
    param pred: prediction, tensor
    param gt: gt, tensor
    return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)            #(1,1) #(预测为1,实际为1)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)#(0,0) #(预测为0,实际为0)
    fp = ((1 - gt) * pred).sum().to(torch.float32)      #(1,0)
    fn = (gt * (1 - pred)).sum().to(torch.float32)      #(0,1)
    # TP (True  Positive）真正例，即模型正确预测为正类别的样本数。
    # TN (True  Negative）真反例，即模型正确预测为负类别的样本数。
    # FP (False Positive）假正例，即模型错误预测为正类别的样本数（实际为负类别但被预测为正类别）。
    # FN (False Negative）假反例，即模型错误预测为负类别的样本数（实际为正类别但被预测为负类别）


    epsilon = 1e-7
    # print("fp",fp)
    # print("fn",fn)
    # print("tp",tp)
    # print("tp",tn)
    precision = tp / (tp + fp + epsilon) # 精确率：预测为T的这些样本中，实际为T的比例 (预测T的正确率)
    recall = tp / (tp + fn + epsilon)    # 召回率：实际为T的这些样本中，预测为T的比例 (实际T的召回率)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    # F1分数（F1 Score）是精确率和召回率的调和平均数，用于衡量一个分类模型的性能。
    Acc = (tp+tn)/(tp+fn+tn+fp) # accuracy：准确率 (全预测正确率)
    Sp = tn/(tn+fp+ epsilon) # 特异性： 预测为F的这些样本中，实际为F的比例 (预测F的正确率)
    jc_score = jc(pred,gt) #Jaccard系数: 重叠度
    AUC = calAUC(pred_con, gt)
    AUC2 = calAUC(pred, gt)
    return f1_score * 100, precision * 100, recall * 100, Sp * 100, Acc * 100, jc_score *100, AUC, AUC2


if __name__ == '__main__':
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    pred = np.array([0.1, 0.4, 0.5, 0.6, 0.35, 0.8, 0.9, 0.4, 0.25, 0.5])
    print("sklearn auc:", roc_auc_score(y,pred))
    print("my auc calc by area:", calAUC(pred,y))
    print("my auc calc by prob:", calAUC(pred,y))