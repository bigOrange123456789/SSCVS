import os

import torch

from config import config

import numpy as np
from PIL import Image

from lib.ConnectivityAnalyzer import ConnectivityAnalyzer
from utils.evaluation_metric import compute_allRetinal
# import csv
# def create_csv(path, csv_head):
#     with open(path, 'w', newline='') as f:
#         csv_write = csv.writer(f)
#         # csv_head = ["good","bad"]
#         csv_write.writerow(csv_head)

# def write_csv(path, data_row):
#     # path  = "aa.csv"
#     with open(path, 'a+', newline='') as f:
#         csv_write = csv.writer(f)
#         # data_row = ["1","2"]
#         csv_write.writerow(data_row)
from lib.csv_lib import create_csv,write_csv,getPath_csv

class Predictor():
    def __init__(
        self,
        Segment_model,
        dataloader_val,
        dataloader_supervised,
        dataloader_unsupervised,
        criterion
    ):
        self.Segment_model=Segment_model
        self.dataloader_val=dataloader_val
        self.dataloader_sup=dataloader_supervised
        self.dataloader_unsup=dataloader_unsupervised
        self.criterion=criterion # DiceLoss()

        # folder_path = os.path.join('logs', config.logname + '.log')
        # try:# 使用 os.makedirs 创建文件夹，如果父文件夹不存在也会一并创建
        #     os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 表示如果文件夹已存在，不会抛出异常
        # except OSError as error:
        #     print(f"创建文件夹 '{folder_path}' 时出错: {error}")
        folder_path = getPath_csv()
        self.val_score_path = folder_path + '/' + 'val_train_f1.csv'
        csv_head = ["epoch", "total_loss", "f1", "AUC", "AUC2", "pr", "recall", "Acc", "Sp", "JC","Dice","Dice2"]
        create_csv(self.val_score_path, csv_head)

        if False:  # 加载保存的状态字典
            self.__loadParm('logs/best_Segment.pt')

    def __loadParm(self,checkpoint_path):
        # checkpoint_path = 'logs/best_Segment.pt'  # os.path.join(cls.logpath, 'best_Segment.pt')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 如果模型是在GPU上训练的，这里指定为'cpu'以确保兼容性
        self.Segment_model.load_state_dict(checkpoint['state_dict'])  # 提取模型状态字典并赋值给模型

    def __inference(self, loader , path ) :
        if not path==None:
            os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            self.Segment_model.module.eval()
        else:  # 将模型设置为评价模式
            self.Segment_model.eval()
        with torch.no_grad():  # 不进行梯度计算
            val_sum_Dice = 0
            val_sum_Dice2 = 0
            val_sum_f1 = 0
            val_sum_pr = 0
            val_sum_re = 0
            val_sum_sp = 0
            val_sum_acc = 0
            val_sum_jc = 0
            val_sum_AUC = 0
            val_sum_AUC2 = 0

            for val_idx, minibatch in enumerate(loader):
                val_imgs = minibatch['img']  # 图片的梯度数据
                val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
                result = self.Segment_model(val_imgs, mask=None, trained=False, fake=False)
                val_pred_sup_l, sample_set_unsup = result["pred"], result["sample_set"]
                if not path == None:
                    val_img_name = minibatch['img_name']  # 图片名称
                    if config.onlyMainObj: #True:
                        val_pred_sup_l = ConnectivityAnalyzer(val_pred_sup_l).mainObj
                    else:
                        val_pred_sup_l = torch.where(val_pred_sup_l > 0.5, torch.ones_like(val_pred_sup_l),
                             torch.zeros_like(val_pred_sup_l))

                    val_pred_sup_l = val_pred_sup_l * 255

                    # 将tensor转换为numpy数组，并调整形状以匹配PIL的输入要求（N, H, W）
                    images_np = val_pred_sup_l.to('cpu').numpy().squeeze(axis=1).astype(np.uint8)
                    # 保存每张图片到本地文件
                    for i, image in enumerate(images_np):
                        # 使用PIL创建图像对象，并保存为灰度图
                        img_pil = Image.fromarray(image, mode='L')  # 'L'模式表示灰度图
                        # img_pil.save("logs/"+val_img_name[i])  # 保存图片，文件名可以根据需要调整
                        # img_pil.save(os.path.join('logs', config.logname + ".log", "inference", val_img_name[i]))
                        img_pil.save(os.path.join(path, val_img_name[i]))
                else:
                    val_gts = minibatch['anno_mask']  # 手工标签
                    val_gts = val_gts.cuda(non_blocking=True)

                    max_l = torch.where(val_pred_sup_l >= 0.5, 1, 0)
                    val_max_l = max_l.float()  # 1.<-1; 0.<-0;
                    val_Dice = self.criterion(val_pred_sup_l, val_gts)  # 监督损失
                    val_Dice2 = self.criterion(val_max_l, val_gts)  # 监督损失

                    val_f1, val_precision, val_recall, val_Sp, val_Acc, val_jc, val_AUC, val_AUC2 = compute_allRetinal(val_max_l,
                                                                                                             val_pred_sup_l,
                                                                                                             val_gts)
                    val_sum_Dice += val_Dice.item() #Dice
                    val_sum_Dice2 += val_Dice2.item()  # Dice
                    val_sum_f1 += val_f1
                    val_sum_pr += val_precision
                    val_sum_re += val_recall
                    val_sum_AUC += val_AUC
                    val_sum_AUC2 += val_AUC2
                    val_sum_sp += val_Sp
                    val_sum_acc += val_Acc
                    val_sum_jc += val_jc
        if path == None:
            val_mean_f1 = val_sum_f1 / len(loader)
            val_mean_pr = val_sum_pr / len(loader)
            val_mean_re = val_sum_re / len(loader)
            val_mean_AUC = val_sum_AUC / len(loader)
            val_mean_AUC2 = val_sum_AUC2 / len(loader)
            val_mean_acc = val_sum_acc / len(loader)
            val_mean_sp = val_sum_sp / len(loader)
            val_mean_jc = val_sum_jc / len(loader)
            val_mean_Dice = val_sum_Dice / len(loader)
            val_mean_Dice2 = val_sum_Dice2 / len(loader)
            return val_mean_f1, val_mean_pr, val_mean_re, val_mean_AUC, val_mean_AUC2, val_mean_acc, val_mean_sp, val_mean_jc,val_mean_Dice,val_mean_Dice2

    def lastInference(self) :
        pathParam = os.path.join('logs', config.logname + ".log", "best_Segment.pt")
        self.__loadParm(pathParam)
        path = os.path.join('logs', config.logname + ".log", "inference")
        self.__inference(self.dataloader_val, path)
    def nextInference(self) :
        path = os.path.join('logs', config.logname + ".log", "unsup_temp")
        self.__inference(self.dataloader_unsup, path)

    def showInput(self):
        path = os.path.join('logs', config.logname + ".log", "liot")
        os.makedirs(path, exist_ok=True)
        # loader = self.dataloader_unsup
        loader = self.dataloader_sup
        with torch.no_grad():  # 不进行梯度计算
            for val_idx, minibatch in enumerate(loader):
                val_img_name = minibatch['img_name']  # 图片名称
                val_imgs = minibatch['img_copy']  # 图片的梯度数据
                val_imgs = val_imgs.cuda(non_blocking=True)  # NCHW
                # print("val_imgs",val_imgs.shape)
                # print("val_imgs[0, 0, :, :]",val_imgs[0, 0, :, :])

                val_imgs_t = minibatch['img_test']  # 图片的梯度数据
                # val_imgs_t = minibatch['img_copy']  # 图片的梯度数据
                # val_imgs_t =val_imgs_t*255

                # gray=minibatch['img']
                # print("gray",gray.shape,type(gray))

                # val_imgs = val_imgs * 255

                # 将tensor转换为numpy数组，并调整形状以匹配PIL的输入要求（N, H, W）
                # images_np = val_imgs[:,0,:,:].to('cpu').numpy()#.squeeze(axis=1).astype(np.uint8)
                images_np = val_imgs[:, 0, :, :].to('cpu').numpy().astype(np.uint8)
                images_np_t = val_imgs_t[:, 0, :, :].to('cpu').numpy().astype(np.uint8)
                # print("images_np_t:",images_np_t)
                # print("images_np_t[0,:,:]",images_np_t[0,:,:])
                # 保存每张图片到本地文件
                for i, image in enumerate(images_np):
                    # 使用PIL创建图像对象，并保存为灰度图
                    img_pil = Image.fromarray(image, mode='L')  # 'L'模式表示灰度图
                    img_pil.save(os.path.join(path, str(i)+"input.png"))

                    img_pil = Image.fromarray(images_np_t[i,:,:], mode='L')  # 'L'模式表示灰度图
                    img_pil.save(os.path.join(path, str(i)+"test.png" ))
                exit(0)

    # def evaluate(self,epoch, Segment_model, predict_Discriminator_model, val_target_loader, criterion):
    def evaluate(self, epoch, train_total_loss):
        Segment_model=self.Segment_model
        # val_target_loader=self.dataloader_val

        '''
        epoch,                      已完成的批次数     0
        Segment_model,              分割模型
        val_target_loader,          验证集加载器
        criterion,                  评价标准           DiceLoss() type=<utils.loss_function.DiceLoss>
        '''
        if torch.cuda.device_count() > 1:
            Segment_model.module.eval()
        else:  # 将模型设置为评价模式
            Segment_model.eval()
        with torch.no_grad():  # 不进行梯度计算

            val_mean_f1, val_mean_pr, val_mean_re, val_mean_AUC, val_mean_AUC2, val_mean_acc, val_mean_sp, val_mean_jc,val_mean_Dice,val_mean_Dice2 =\
                self.__inference( self.dataloader_val , None )

            data_row_f1score = [str(epoch), str(train_total_loss), str(val_mean_f1.item()), str(val_mean_AUC), str(val_mean_AUC2),
                                str(val_mean_pr.item()), str(val_mean_re.item()), str(val_mean_acc),
                                str(val_mean_sp), str(val_mean_jc),str(val_mean_Dice),str(val_mean_Dice2)]
            print("val_mean_f1", val_mean_f1.item())
            print("val_mean_AUC", val_mean_AUC)
            print("val_mean_pr", val_mean_pr.item())
            print("val_mean_re", val_mean_re.item())
            print("val_mean_acc", val_mean_acc.item())
            write_csv(self.val_score_path, data_row_f1score)
            return val_mean_f1


