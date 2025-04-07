# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import time
import numpy as np
from easydict import EasyDict as edict

import sys
import json
json_file_path = 'config.json'
# if len(sys.argv) > 1:
#     json_file_path = sys.argv[1]
# print('json_file_path:', json_file_path)
C_ = json.load(open(json_file_path, 'r', encoding='utf-8') )

C = edict()
config = C
cfg = C

C.seed = C_['seed']#12345  #3407

# remoteip = os.popen('pwd').read()
if os.getenv('volna') is not None:
    C.volna = os.environ['volna']
else:
    C.volna = '/data/sty/Unsupervised/' # the path to the data dir.

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'TorchSemiSeg'
C.datapath = C_["datapath"] #"./Data/XCAD"
C.benchmark = 'XCAD_LIOT'#XCAD_LIOT
C.logname = C_["logname"] #'test_XCADmodel'
#next experiment memory
C.log_dir = "/data/sty/Unsupervised_dxh"
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.dataset_path = C.volna + "Data/CrackTree"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_img_root_folder = C.dataset_path + "/train/img"
C.train_fakegt_root_folder = C.dataset_path + "/train/fake_vessel"
C.val_img_root_folder = C.dataset_path + "/test/img"
C.val_gt_root_folder = C.dataset_path + "/test/gt"
C.pretrained_model = C.volna +  'DATA/pytorch-weight/resnet50_v1c.pth'

# model path
C.model_weight = '/data/sty/Unsupervised_dxh/logs/contrast_repeatnce4_negall_Nce_allpixel_noise.log/best_Segment.pt'

"""Path Config"""


''' Experiments Setting '''
C.labeled_ratio = 1
C.train_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(C.dataset_path, "config_new/subset_train/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))
C.eval_source = osp.join(C.dataset_path, "config_new/val.txt")
C.test_source = osp.join(C.dataset_path, "config_new/test.txt")
C.demo_source = osp.join(C.dataset_path, "config_new/demo.txt")

C.is_test = False
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

C.cps_weight = 6


''' Our New Setting(我们自己的设置) '''
# C.idxBatchPrint = C_["idxBatchPrint"]#不再每个idx都输出
#1.是否使用通过体积渲染将血管添加到背景上
C.vessel3D = C_["vessel3D"] #使用合成的3D血管/使用合成的2D血管
C.dePhase=C_["dePhase"]#将合成的3D血管添加到背景图之前，需要对背景图去噪声(傅里叶相位) #{0不去噪，1去傅里叶相位，2去傅里叶相位+去对称}
#2.是否使用不平衡损失
C.ASL = True #C_["ASL"] #不平衡损失/加权BCE
# gamma_neg=0, gamma_pos=0 的时候应该就相当于C.ASL = False
C.gamma_neg=C_["ASL"]["gamma_neg"]
C. gamma_pos=C_["ASL"]["gamma_pos"]
#3.是否使用大间隔对比学习
C.marginInfoNCE = C_['marginInfoNCE']
#4.是否使用伪标签机制
C.pseudo_label = C_["pseudo_label"]
C.onlyMainObj  = C_["onlyMainObj"]
#5.是否使用连通性损失
C.connectivityLoss = C_["conn"]["weight"]>0 # C_["connectivityLoss"]
C.connectivityLossType = C_["conn"]["type"] # C_["connectivityLossType"]
#6.设置神经网络模型输入的图片格式
C.inputType = C_["inputType"]
#7.使用TeacherStudent架构
C.useEMA = C_["cons"]["useEMA"] #C_["useEMA"]
######################################################################################################
C.adv = C_["adv"]
C.cons = C_["cons"]
C.seg = C_["seg"]
C.contrast = C_["contrast"]
C.conn = C_["conn"]
C.datapathTrain = C_["datapathTrain"]
######################################################################################################

"""Image Config"""
C.num_classes = 1
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.image_height = 256
C.image_width = 256
C.num_train_imgs = 150 // C.labeled_ratio #XCAD
C.num_eval_imgs = 126#XCAD
C.num_unsup_imgs = 1621 #XCAD

"""Train Config"""
C.lr = 0.01
C.lr_D = 0.001
# C.batch_size = 8
C.batch_size = C_["batch_size"] # 4
C.batch_size_val = C_["batch_size_val"] # 1
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.state_epoch = 0
C.nepochs = 1000
C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)
C.cold_start = 0
C.niters_per_epoch = C.max_samples // C.batch_size

C.nworker = 8
C.img_mode = 'crop'
C.img_size = 256
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_base_size = 800
C.eval_crop_size = 800

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50
C.warm_up_epoch = 0

