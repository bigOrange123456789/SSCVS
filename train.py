# from __future__ import division
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import config

from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from Datasetloader.dataset import CSDataset
from common.logger import Logger
from utils.loss_function import DiceLoss

from lib.ModelDiscriminate import ModelDiscriminate
from lib.ModelSegment import ModelSegment
from lib.Predictor import Predictor
from lib.Trainer import Trainer

def main():
    # os.getenv('debug'): None
    if os.getenv('debug') is not None:
        is_debug = os.environ['debug']
    else:
        is_debug = False
    parser = argparse.ArgumentParser()
    os.environ['MASTER_PORT'] = '169711' #“master_port”的意思是主端口

    args = parser.parse_args()
    cudnn.benchmark = True #benchmark的意思是基准
    # set seed
    seed = config.seed  # 12345
    torch.manual_seed(seed) # manual_seed的意思是人工种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("Begin Dataloader.....")

    CSDataset.initialize(datapath=config.datapath)
    print('config.datapath:',config.datapath)#"./Data/XCAD"

    dataloader_supervised,_ = CSDataset.build_dataloader(config.benchmark,  # XCAD_LIOT # benchmark的本译是基准
                                                       config.batch_size, # 4
                                                       config.nworker,    # 8
                                                       'train',
                                                       config.img_mode,   # crop
                                                       config.img_size,   # 256
                                                       'supervised')
    # 有监督使用的数据是什么？gray中为血管的灰度图、gt中为标签。
    dataloader_unsupervised,dataset_unsupervised = CSDataset.build_dataloader(config.benchmark,  # XCAD_LIOT
                                                         config.batch_size, # 4
                                                         config.nworker,    # 8
                                                         'train',
                                                         config.img_mode,   # crop
                                                         config.img_size,   # 256
                                                         'unsupervised')
    # 无监督使用的应该是img文件夹中的原始图片。

    dataloader_val,_ = CSDataset.build_dataloader(config.benchmark,       # XCAD_LIOT
                                                config.batch_size_val,  # 1
                                                config.nworker,         # 8
                                                'val',
                                                'same',
                                                None,
                                                'supervised')
    # 无监督使用的应该是img文件夹中的原始图片。
    print("Dataloader.....")
    criterion = DiceLoss()  # try both loss BCE and DICE # 尝试损失BCE和DICE
    # dice(x,y) = 1 - 2 * (x*y) / (x+y) # 相同为-1,不同为1

    # define and init the model # 定义并初始化模型
    # Single or not single # 单个或非单个
    BatchNorm2d = nn.BatchNorm2d # <BatchNorm2d> #BN会重置均值和方差，新的均值和新的方差都是可学习的参数
    # Segment_model = Single_contrast_UNet(4, config.num_classes) # config.num_classes=1
    if config.inputType=="Origin":
        n_channels = 1
    elif config.inputType=="LIOT":
        n_channels = 4
    elif config.inputType == "NewLIOT":
        n_channels = 4
    elif config.inputType == "NewLIOT2":
        n_channels = 4
    else:
        print("配置文件中的inputType参数不合法!")
        exit(0)
    # Segment_model = Single_contrast_UNet(n_channels, config.num_classes) # 我猜BN不放在Segment_model中的原因是：训练和评估这两种模式在使用的时候会有差异
    Segment_model = ModelSegment(n_channels, config.num_classes)
    if config.useEMA:
        Segment_model_EMA = ModelSegment(n_channels, config.num_classes)
    else:
        Segment_model_EMA = None

    init_weight(Segment_model.business_layer, nn.init.kaiming_normal_,
                # nn.init.kaiming_normal_: <function kaiming_normal_>
                BatchNorm2d,        # BatchNorm2d: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
                config.bn_eps,      # config.bn_eps: 1e-05
                config.bn_momentum, # config.bn_momentum: 0.1
                mode='fan_in', nonlinearity='relu')
    # define the learning rate
    base_lr = config.lr      # 0.04 # 学习率
    base_lr_D = config.lr_D  # 0.04 # dropout?

    predictor = Predictor(Segment_model, dataloader_val, dataloader_supervised,dataloader_unsupervised, criterion)
    # predictor.showInput()#测试代码

    params_list_l = []
    params_list_l = group_weight(
        params_list_l, #一个list对象 #用于存储tensor对象
        Segment_model.backbone, # 分割网络的主干
        BatchNorm2d,    # BatchNorm2d: <class 'torch.nn.modules.batchnorm.BatchNorm2d'>
        base_lr)        # base_lr: 0.01
    if hasattr(Segment_model, 'learnable_scalar'): #用于优化对比学习的一个边缘间隔margin参数
        params_list_l.append(dict(params=Segment_model.learnable_scalar, lr=base_lr))
    # optimizer for segmentation_L   # 分割优化器_L
    print("config.weight_decay", config.weight_decay)
    optimizer_l = torch.optim.SGD(params_list_l,#分割网络中的全部参数
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    # predict_Discriminator_model = PredictDiscriminator(num_classes=1)
    predict_Discriminator_model = ModelDiscriminate(num_classes=1)
    init_weight(predict_Discriminator_model, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    optimizer_D = torch.optim.Adam(predict_Discriminator_model.parameters(),#判别器中的全部参数
                                   lr=base_lr_D, betas=(0.9, 0.99))

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch  # nepochs=137  niters=C.max_samples // C.batch_size
    print("total_iteration", total_iteration)
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    lrD_policy = WarmUpPolyLR(base_lr_D, config.lr_power, total_iteration,
                              config.niters_per_epoch * config.warm_up_epoch)

    # 有1个cuda 。torch.cuda.device_count()=1
    if torch.cuda.device_count() > 1:
        Segment_model = Segment_model.cuda()
        Segment_model = nn.DataParallel(Segment_model)
        if Segment_model_EMA!=None:
            Segment_model_EMA = Segment_model_EMA.cuda()
            Segment_model_EMA = nn.DataParallel(Segment_model_EMA)
        predict_Discriminator_model = predict_Discriminator_model.cuda()
        predict_Discriminator_model = nn.DataParallel(predict_Discriminator_model)
        # Logger.info('Use GPU Parallel.')
    elif torch.cuda.is_available():
        print("cuda_is available")
        Segment_model = Segment_model.cuda() # 分割模型
        if Segment_model_EMA != None:
            Segment_model_EMA = Segment_model_EMA.cuda()
        predict_Discriminator_model = predict_Discriminator_model.cuda() # 预测判别模型
    else:
        Segment_model = Segment_model
        if Segment_model_EMA != None:
            Segment_model_EMA = Segment_model_EMA
        predict_Discriminator_model = predict_Discriminator_model

    best_val_f1 = 0
    Logger.initialize(config, training=True)
    trainer = Trainer(Segment_model,Segment_model_EMA, predict_Discriminator_model, dataloader_supervised, dataloader_unsupervised,
                optimizer_l, optimizer_D, lr_policy, lrD_policy, criterion, total_iteration)
    # inference(Segment_model, dataloader_val)
    # predictor.showInput()
    # exit(0)
    for epoch in range(config.state_epoch, config.nepochs): # 从state_epoch到nepochs-1 # 按照预先设定的回合数量执行，不会提前中止
        '''train_total_loss=0'''

        train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss, train_loss_dice, train_loss_ce, train_loss_contrast \
            =trainer.train(epoch, dataset_unsupervised.isFirstEpoch)
        print("train_seg_loss:{},train_loss_Dtar:{},train_loss_Dsrc:{},train_loss_adv:{},train_total_loss:{},train_loss_contrast:{}".format(
                train_loss_seg, train_loss_Dtar, train_loss_Dsrc, train_loss_adv, train_total_loss,train_loss_contrast))
        print("train_loss_dice:{},train_loss_ce:{}".format(train_loss_dice, train_loss_ce))

        val_mean_f1 = predictor.evaluate(epoch, train_total_loss)
        if val_mean_f1 > best_val_f1: # F1分数是精确率和召回率的调和平均数
            best_val_f1 = val_mean_f1
            Logger.save_model_f1_S(Segment_model, epoch, val_mean_f1, optimizer_l) #保存到best_Segment.pt中
            Logger.save_model_f1_T(predict_Discriminator_model, epoch, val_mean_f1, optimizer_D) #保存到best_Dis.pt中

        if config.pseudo_label:
            predictor.nextInference()
            dataset_unsupervised.isFirstEpoch=False #已经保存了伪标签数据

    predictor.lastInference()# inference(Segment_model, dataloader_val)

if __name__ == '__main__':
    main()

'''
    2. 训练脚本
    export PATH="~/anaconda3/bin:$PATH"
    source activate FreeCOS
    python train_DA_contrast_liot_finalversion.py 
    #(CUDA_VISIBLE_DEVICES=0 python train_DA_contrast_liot_DRIVE_finalversion.py for DRIVE)
'''
