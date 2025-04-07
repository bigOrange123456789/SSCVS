from config import config
import random

import torch
import os.path
import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as F

from torchvision import transforms
from torch.utils import data
from Datasetloader.torch_LIOT import trans_liot, trans_NewLiot, trans_NewLiot2, trans_liot_region, \
    trans_liot_region_stride, trans_liot_differentsize
# trans_list NCHW
import cv2

# 设置随机数种子
random.seed(config.seed)
np.random.seed(42)


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img  # .cpu().numpy()
    trg_img_np = trg_img  # .cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    return src_in_trg


class DatasetXCAD_aug(data.Dataset):

    def __init__(self, benchmark, datapath, split, img_mode, img_size, supervised):
        super(DatasetXCAD_aug, self).__init__()
        self.isFirstEpoch = True  # 还没有保存了伪标签数据
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.benchmark = benchmark  # benchmark = XCAD_LIOT
        assert self.benchmark == 'XCAD_LIOT'
        self.img_mode = img_mode  # 测试时为same,训练时为crop (译为裁减)
        assert img_mode in ['crop', 'same', 'resize']  # 有三种图片模式：裁减、相似、放缩
        self.img_size = img_size  # int # 训练时图片大小为256、测试时为None
        self.supervised = supervised  # 训练集包括有监督和无监督两部分、测试集为有监督

        self.datapath = datapath  # ./Data/XCAD
        datapathTrain = config.datapathTrain
        if self.supervised == 'supervised':
            if self.split == 'train':  # 有监督的训练
                '''
                self.img_path = os.path.join(datapath, 'train','fake_grayvessel_width')#./Data/XCAD/train/fake_grayvessel_width #合成血管
                self.img_path_3D = os.path.join(datapath, 'train', 'vessel3D_lzc')  # ./Data/XCAD/train/vessel3D_lzc
                self.background_path    = os.path.join(datapath,'train', 'img')           #./Data/XCAD/train/img                   #真实图片
                self.background_path_3D = os.path.join(datapath, 'train', 'bg_lzc')  # ./Data/XCAD/train/img
                self.ann_path    = os.path.join(datapath, 'train','fake_gtvessel_width')  #./Data/XCAD/train/fake_gtvessel_width   #合成标签
                '''
                self.img_path = os.path.join(datapath, 'train',
                                             datapathTrain["vessel"])  # ./Data/XCAD/train/fake_grayvessel_width #合成血管
                self.img_path_3D = os.path.join(datapath, 'train',
                                                datapathTrain["vessel_3D"])  # ./Data/XCAD/train/vessel3D_lzc
                self.background_path = os.path.join(datapath, 'train', datapathTrain[
                    "bg"])  # ./Data/XCAD/train/img                   #真实图片
                self.background_path_3D = os.path.join(datapath, 'train',
                                                       datapathTrain["bg_3D"])  # ./Data/XCAD/train/img
                self.ann_path = os.path.join(datapath, 'train',
                                             datapathTrain["label"])  # ./Data/XCAD/train/fake_gtvessel_width   #合成标签
                self.ann_path_3D = os.path.join(datapath, 'train', datapathTrain["label_3D"])

                if not config.vessel3D:
                    self.img_metadata = os.listdir(
                        self.img_path)  # self.load_metadata_supervised()  #train_fakevessel.txt
                    self.background_metadata = os.listdir(
                        self.background_path)  # self.load_metadata_background()  #train_backvessel.txt
                else:
                    self.img_metadata = os.listdir(self.img_path_3D)
                    self.background_metadata = os.listdir(self.background_path_3D)
            else:  # 有监督的测试
                self.img_path = os.path.join(datapath, 'test', 'img')  # ./Data/XCAD/test/img #真实数据
                self.ann_path = os.path.join(datapath, 'test', 'gt')  # ./Data/XCAD/test/gt  #人工标签
                self.img_metadata = self.load_metadata_testsupervised()  # test_img.txt
        else:  # 无监督的训练
            self.img_path = os.path.join(datapath, 'train', datapathTrain["img_unsup"])  # ./Data/XCAD/train/img
            self.ann_path = os.path.join('logs', config.logname + ".log", "unsup_temp")  # 伪标签
            self.img_metadata = os.listdir(self.img_path)  # self.load_metadata_background()  #train_backvessel.txt
        self.norm_img = transforms.Compose([
            transforms.ToTensor()  # 将数据由HWC255格式 转换为CHW0～1的格式
        ])
        '''
        transforms.Compose([...])：
            Compose是一个函数，它接受一个列表作为输入，这个列表包含了多个图像转换操作。
            Compose的作用是将这些转换操作按顺序应用到图像上。
        transforms.ToTensor()：
            ToTensor是一个转换操作，它将PIL图像或者一个形状为HWC（高度、宽度、通道数）的NumPy ndarray（数据类型为uint8）
            转换成形状为CHW的FloatTensor，
            并且把数值范围从[0, 255]缩放到[0.0, 1.0]。
            这个转换是图像预处理中非常常见的一步，因为它使得图像数据适合作为神经网络模型的输入。
        self.norm_img = ...：
            这行代码将Compose函数返回的转换组合（即包含ToTensor操作的组合）赋值给self.norm_img。
            在这里，self.norm_img并不是直接存储图像数据，而是存储了一组转换操作。
            这个组操作可以在后续的数据加载和预处理阶段被应用到图像数据上。
        '''
        if self.img_mode == 'resize':
            self.resize = transforms.Resize([img_size, img_size], interpolation=Image.NEAREST)
        else:
            self.resize = None

    def __len__(self):
        return len(self.img_metadata)  # img_metadata中存储了所有文件的名称

    def __getitem__(self, index):
        img_name = self.img_metadata[
            index]  # get the name of fakevessel(train-supervised), img(test-supervised), img(train-unsupervised)
        supervised_pseudo = config.pseudo_label and self.isFirstEpoch == False
        img_Perturbation = None
        if self.supervised == 'supervised' and self.split == 'train':  # 有监督的训练
            idx_background = np.random.randint(len(self.background_metadata))  # background(train-supervised)
            background_name = self.background_metadata[idx_background]  # 随机抽取一张背景图(造影图)
            '''
            self.background_metadata 中为 train_backvessel.txt 
                原本是img文件夹中的子集文件名
                但由于img文件夹与bg_lzc文件夹中的文件名是一致的，所以也可以作为bg_lzc文件夹的子集文件名
            '''
            if config.vessel3D:  # LZC:这部分新增的代码用于读取3D血管数据
                img, anno_mask, org_img_size, img_Perturbation = self.load_frame_fakevessel_3D(img_name,
                                                                                               background_name)
                # (合成图像、合成标签)<-(合成血管图、真实造影图)
                # print("img",type(img))
                # print("anno_mask",type(anno_mask))
                # exit(0)
            else:  # 之前的获取2D血管数据的代码
                img, anno_mask, org_img_size = self.load_frame_fakevessel_gaussian(img_name,
                                                                                   background_name)  # (合成图像、合成标签)<-(合成血管图、真实造影图)
        elif self.supervised == 'supervised' and self.split != 'train':  # 有监督的验证
            img, anno_mask, org_img_size = self.load_frame_aff(img_name)  # 返回：造影图、人工标签、尺寸
            # print("1supervised",type(img),type(anno_mask),img.shape,anno_mask.shape)
            # print("1supervised",anno_mask.shape)
        else:  # 无监督的训练
            img, org_img_size = self.load_frame_unsupervised(img_name)  # 返回：造影图、尺寸
            anno_mask = None  # 无标签
            if supervised_pseudo:
                anno_mask = self.read_mask(img_name)
                # print(type(anno_mask))
                # print(img)
                # print("1 unsupervised",img.shape)
                # print("1 unsupervised",type(img),type(anno_mask),anno_mask.shape)
                # print("1 unsupervised",anno_mask.shape)

        if self.split == 'train' and self.supervised == 'supervised':  # 有监督 or 伪监督
            img, anno_mask, img_Perturbation = self.augmentation_aff(img, anno_mask, img_Perturbation)  # 翻转、旋转、调色
        elif self.split == 'train' and self.supervised != 'supervised':  # 无监督的训练
            if not supervised_pseudo:  # 伪标签数据不进行增强（因为对伪标签数据进行增强会导致BUG）
                img, anno_mask = self.augmentation_unsupervised(img, anno_mask)  # 翻转、旋转、调色
        # if anno_mask!=None:
        #     img, anno_mask = self.augmentation_aff(img, anno_mask) #翻转、旋转、调色
        #     # print("1.5 ",img.shape,anno_mask.shape)
        # else:
        #     img, anno_mask = self.augmentation_unsupervised(img,anno_mask) #翻转、旋转、调色

        # self.img_mode: crop
        if self.img_mode == 'resize' and self.split == 'train':
            img = self.resize(img)
            if img_Perturbation != None:
                img_Perturbation = self.resize(img_Perturbation)
            if anno_mask != None:
                anno_mask = self.resize(anno_mask)
        elif self.img_mode == 'crop' and self.split == 'train':  # 只用训练的时候需要裁减、因为训练前的旋转操作会放大图片
            i, j, h, w = self.get_params(img, (self.img_size, self.img_size))
            # print("i, j, h, w",i, j, h, w)
            img = F.crop(img, i, j, h, w)
            if img_Perturbation != None:
                img_Perturbation = F.crop(img_Perturbation, i, j, h, w)
            if anno_mask != None:
                anno_mask = F.crop(anno_mask, i, j, h, w)
                # print(anno_mask.shape, i, j, h, w)
        else:
            pass


        img_gray = self.norm_img(np.array(img))  # 由前文知norm_img作用为: 将数据由HWC255 转换为CHW0～1
        img_copy = 255.*img_gray#self.norm_img(np.array(img_Perturbation))

        # img:      (256, 256) <PIL.Image.Image>
        # img_gray: [1, 256, 256] <torch.Tensor>
        # [1, 256, 256] <- (256, 256) # Tensor <- ndarray

        def getInput(img):#if True:#
            # img_test = img_gray
            if config.inputType == "Origin":
                img = np.asarray(img)
                img = img[None, :, :]  # 新增加一个维度
            elif config.inputType == "LIOT":  # LIOT part
                img = trans_liot(img)  # (4,256,256) <- (256,256) #类似于梯度的计算
            elif config.inputType == "NewLIOT":  # LIOT
                img = trans_NewLiot(img)
            elif config.inputType == "NewLIOT2":  # LIOT
                img = trans_NewLiot2(img)
            else:
                print("配置文件中的inputType参数不合法！")
                exit(0)
            img_test = np.copy(img) #用于测试

            img = img.transpose((1, 2, 0))  # (256,256,4) <- (4,256,256)
            img = self.norm_img(img)  # [4,256,256] <- (256,256,4) #由HWC255格式、转换为CHW0～1的格式
            # [4, 256, 256] <- (256, 256, 4) # Tensor <- ndarray
            # img = img_gray
            return (img - torch.mean(img)) / torch.std(img) ,img_test # (x-均值)/标准差 # 将均值转换为0、标准差转换为1

        img,_ = getInput(img)
        if img_Perturbation != None:
            img_Perturbation,img_test = getInput(img_Perturbation)

        if self.supervised == 'supervised':  # 有监督
            # print("gt_unique", torch.unique(anno_mask))
            batch = {
                'img_name': img_name,  # 图片名称
                'img': img,  # (梯度)图片数据
                'anno_mask': anno_mask,  # 标签数据
                'gray': img_gray  # (原始)图像数据
                # 'img_Perturbation': img_Perturbation  # 扰动图像
            }
            if img_Perturbation != None:
                batch["img_Perturbation"] = img_Perturbation  # 扰动图像
            if False:
                batch["img_test"] = img_test
                batch["img_copy"] = img_copy
            # print("2supervised",img.shape,anno_mask.shape)
            # print("2.supervised",anno_mask.shape)
            return batch  # 实际训练过程中只用 img、anno_mask 这两个值被用到
        else:  # 无监督
            batch = {
                'img_name': img_name,  # 图片名称
                'img': img  # 图片数据
            }
            if config.pseudo_label and self.isFirstEpoch == False:
                batch = {
                    'img_name': img_name,  # 图片名称
                    'anno_mask': anno_mask,  # 伪标签数据
                    'img': img  # 图片数据
                }
                # print("2supervised",img.shape,anno_mask.shape)
                # print("2.unsupervised",anno_mask.shape,self.img_mode == 'crop' and self.split == 'train',h,w)
            # batch["img_test"]=img_test
            # batch["img_copy"]=img_copy
            return batch

    def augmentation(self, img, anno_mask, anno_boundary, ignore_mask):
        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)
        img = transform_hflip(img)
        anno_mask = transform_hflip(anno_mask)
        anno_boundary = transform_hflip(anno_boundary)
        ignore_mask = transform_hflip(ignore_mask)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)
        img = transform_vflip(img)
        anno_mask = transform_vflip(anno_mask)
        anno_boundary = transform_vflip(anno_boundary)
        ignore_mask = transform_vflip(ignore_mask)

        if np.random.random() > 0.5:
            p = np.random.uniform(-180, 180, 1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)
            anno_mask = transform_rotate(anno_mask)
            anno_boundary = transform_rotate(anno_boundary)
            ignore_mask = transform_rotate(ignore_mask)

        if np.random.random() > 0.5:
            color_aug = transforms.ColorJitter(brightness=[1.0, 2.1], contrast=[1.0, 2.1], saturation=[0.5, 1.5])
            img = color_aug(img)

        return img, anno_mask, anno_boundary, ignore_mask

    def augmentation_aff(self, img, anno_mask, img_Perturbation):  # 有监督数据的数据增强
        # 1.翻转
        p = np.random.choice([0, 1])  # 随机选择一个元素（0或1）
        transform_hflip = transforms.RandomHorizontalFlip(p)  # 定义一个水平翻转操作，p 是这个转换操作被应用的概率。
        img = transform_hflip(img)
        img_Perturbation = transform_hflip(img_Perturbation)
        anno_mask = transform_hflip(anno_mask)  # 若图片翻转，则标签也进行翻转

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)  # 垂直翻转
        img = transform_vflip(img)
        img_Perturbation = transform_vflip(img_Perturbation)
        anno_mask = transform_vflip(anno_mask)

        # 2.旋转 #img = self.to_tensor(img)
        if np.random.random() > 0.5:  # 旋转范围是0～360度，旋转后尺寸会扩大
            p = np.random.uniform(-180, 180, 1)[0]  # 均匀分布从[low, high)内中抽取1个样本
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            '''
            随机角度的旋转
                (p, p) 指定了旋转角度的范围。
                expand=True 时，如果旋转导致图像尺寸超出原始尺寸，则尺寸会被相应扩大，同时中心不变，新创建的像素位置上填充0（黑色）。
                    为False，则旋转后的图像会被裁剪以适应原始图像尺寸，可能会导致图像的一部分被裁剪掉。
            '''
            img = transform_rotate(img)
            img_Perturbation = transform_rotate(img_Perturbation)
            anno_mask = transform_rotate(anno_mask)

        # 3.调色（亮度、对比度、饱和度）
        if np.random.random() > 0.5:
            color_aug = transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.8, 2.1], saturation=[0.5, 1.5])
            # augLOwcon0.8-1.5、0.5-1.5、0.5-2.1 brigntness 0.5-1
            '''
                brightness=[0.5, 1.5]：亮度。
                    这意味着图像可能会变得更暗（乘以小于1的数）或更亮（乘以大于1的数）。
                    y = x * brightness_factor
                contrast=[0.8, 2.1]：对比度。
                    对比度增加会使图像中的颜色差异更加显著，而对比度减少则会使颜色差异变得不那么明显。
                    y = (x - mean) * contrast_factor + mean
                saturation=[0.5, 1.5]：饱和度。
                    饱和度增加会使颜色更加鲜艳，而饱和度减少则会使颜色变得更加灰暗。 
                    转换到HSV（或HSL）色彩空间，然后调整S（饱和度）通道来实现的。
                    new_saturation = original_saturation * saturation_factor
                    在HSL和HSV色彩模型中，饱和度的计算公式为：S=(C/V)×100%。其中，C表示颜色的色相，V表示颜色的亮度。
                    -------------------------------------------------
                    饱和度指的是图像颜色的纯度或鲜艳程度。
                    S=(MaxV-MinV)/MaxV
                        MaxV和MinV分别是R、G、B三个颜色通道中的最大值和最小值。
                    调整饱和度可以通过对图像的每个像素的颜色值相对于灰度颜色进行插值来实现。具体的插值计算方法可能因不同的图像处理库或算法而有所差异。
                此外，尽管ColorJitter还允许调整色调（hue）。
                    转换到HSV（或HSL）色彩空间，然后调整色调（Hue）通道来实现的。
                    new_hue = (original_hue + hue_shift) % 360
            '''
            img = color_aug(img)

        return img, anno_mask, img_Perturbation

    def augmentation_unsupervised(self, img, anno_mask):
        p = np.random.choice([0, 1])
        transform_hflip = transforms.RandomHorizontalFlip(p)  # 水平翻转
        img = transform_hflip(img)

        p = np.random.choice([0, 1])
        transform_vflip = transforms.RandomVerticalFlip(p)  # 垂直翻转
        img = transform_vflip(img)

        if np.random.random() > 0.5:  # 旋转
            p = np.random.uniform(-180, 180, 1)[0]
            transform_rotate = transforms.RandomRotation((p, p), expand=True)
            img = transform_rotate(img)

        if np.random.random() > 0.5:  # 调色
            color_aug = transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.8, 2.1], saturation=[0.5,
                                                                                                       1.5])  # augLOwcon0.8-1.5、0.5-1.5、0.5-2.1 brigntness 0.5-1
            img = color_aug(img)

        return img, anno_mask

    def load_frame(self, img_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        anno_boundary = self.read_boundary(img_name)
        ignore_mask = self.read_ignore_mask(img_name)

        org_img_size = img.size

        return img, anno_mask, anno_boundary, ignore_mask, org_img_size

    def load_frame_aff(self, img_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        org_img_size = img.size

        return img, anno_mask, org_img_size

    def load_frame_fakevessel(self, img_name, background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)

        background_array = np.array(background_img)
        im_src = np.asarray(img, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src, axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.02)
        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA, axis=0)
        img_FDA_Image = Image.fromarray((img_FDA).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_fakevessel_gaussian(self, img_name, background_name):
        img = self.read_img(img_name)  # 读取人工血管图
        anno_mask = self.read_mask(img_name)  # 读取人工血管的标签图
        background_img = self.read_background(background_name)  # 背景图(真实造影图) # <PIL.Image.Image>

        # 1.FDA：在人工血管图中添加背景
        background_array = np.array(background_img)  # <numpy.ndarray> #将图片由'PIL.Image.Image'格式转化为numpy格式
        im_src = np.asarray(img, np.float32)  # <PIL.Image.Image> -- <numpy.ndarray> #转换为NumPy，并且指定类型

        im_trg = np.asarray(background_array, np.float32)  # 转化前后类型都是<numpy.ndarray> (512, 512)
        im_src = np.expand_dims(im_src, axis=2)  # (512, 512) -> (512, 512, 1) #增加一个维度
        im_trg = np.expand_dims(im_trg, axis=2)  # (512, 512) -> (512, 512, 1)

        im_src = im_src.transpose((2, 0, 1))  # (512, 512, 1) -> (1, 512, 512)
        im_trg = im_trg.transpose((2, 0, 1))  # 通过转置操作，改变维度顺序
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.3)  # 源图片是人工血管、目标图片是背景图
        img_FDA = np.clip(src_in_trg, 0, 255.)  # 应该是限制像素的最小值为0、最大值为255
        img_FDA = np.squeeze(img_FDA, axis=0)  # (1, 512, 512) -> (512, 512)
        # 2.高斯模糊
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (13, 13), 0)
        '''
        进行高斯模糊处理。 (注意这里有一个拼写错误，“guassian”应该是“gaussian”)
            img_FDA：原始图像。
            (13, 13)：高斯核的大小。在这个例子中，核的大小是13x13像素。核的大小会影响模糊的程度；较大的核会产生更强烈的模糊效果。
            0：表示σ值在X和Y方向上的标准差。当这个参数为0时，它会根据核的大小自动计算。标准差决定了高斯函数的宽度，从而影响模糊的程度。
        '''
        # 3.添加点噪声
        noise_map = np.random.uniform(-5, 5, img_FDA_guassian.shape)
        '''
        生成一个与img_FDA_guassian图像形状相同的噪声，其中的元素值在-5到5之间均匀分布:
            [-5,5)：抽取样本的范围。
            img_FDA_guassian.shape：输出数组的形状。
        '''
        img_FDA_guassian = img_FDA_guassian + noise_map
        img_FDA_guassian = np.clip(img_FDA_guassian, 0, 255.)
        '''
            img_FDA_guassian [[ ....]] 
            type=<class 'numpy.ndarray'> 
            shape=(512, 512)
        '''

        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')
        '''
        img_FDA_guassian.astype('uint8')：转换为无符号8位整型（uint8）。
            这是必要的，因为PIL库处理图像时通常期望图像数据是以这种格式存储的。
            以确保所有的值都在0到255的范围内（uint8能表示的最小值和最大值）。
        Image.fromarray(...)：这一步使用PIL库的Image.fromarray函数将NumPy数组转换为PIL图像对象。
            这个函数接受一个NumPy数组作为输入，并返回一个PIL图像对象，该对象可以用于进一步的图像处理或保存。
        .convert('L')：这一步将PIL图像对象转换为灰度图像。
            如果原始图像已经是灰度图像，这一步则不会改变图像的内容。
        '''

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_fakevessel_3D(self, img_name, background_name):

        img = self.read_img_3D(img_name)  # 读取3D人工血管图
        # print("img",img.size)
        anno_mask = self.read_mask_3D(img_name)  # 读取人工血管的标签图
        background_img = self.read_background_3D(background_name)  # 背景图(真实造影图) # <PIL.Image.Image>
        # print('background_img',background_img.size)
        # exit(0)

        # 1.将血管添加到背景中：对应元素相称
        # background_array = np.array(background_img)  #背景 <PIL.Image.Image> -> <numpy.ndarray>
        background_array = np.asarray(background_img, np.float32)  # 背景 <PIL.Image.Image> -> <numpy.ndarray>

        def dePhase1(img, MaintainRange):  # 用于去除图片中的傅里叶相位
            if MaintainRange:
                min1 = max(np.min(img), 0.)  # 计算像素的最小值
                ran1 = min(np.max(img), 255.) - min1

            img = np.expand_dims(img, axis=2)  # (512, 512) -> (512, 512, 1)
            img = img.transpose((2, 0, 1))  # 通过转置操作，改变维度顺序

            fft_trg_np = np.fft.fft2(img, axes=(-2, -1))  # 傅里叶变换

            # extract amplitude and phase of both ffts
            amp, pha = np.abs(fft_trg_np), np.angle(fft_trg_np)
            pha = np.zeros_like(pha)

            # mutated fft of source
            fft_src_ = amp * np.exp(1j * pha)

            # 逆傅里叶变换 # get the mutated image
            src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
            src_in_trg = np.real(src_in_trg)  # 从复数数组中提取实部。

            if MaintainRange:
                eps = 1e-8
                img_FDA = src_in_trg
                min2 = np.min(img_FDA)  # 计算像素的最小值
                ran2 = np.max(img_FDA) - min2
                img_FDA = (img_FDA - min2) / (ran2 + eps)
                img_FDA = img_FDA * ran1 + min1
                img_FDA = np.clip(img_FDA, 0, 255.)  # 限制像素的最小值为0、最大值为255
            else:
                img_FDA = np.clip(src_in_trg, 0, 255.)  # 限制像素的最小值为0、最大值为255
            img_FDA = np.squeeze(img_FDA, axis=0)  # (1, 512, 512) -> (512, 512)
            return img_FDA

        def dePhase2(img):  # 去除图片中的傅里叶相位、并通过只取1/4来去除图片对称
            w, h = img.shape  # 使用cv2.resize调整形状为(1024, 1024)，使用双线性插值
            img = cv2.resize(img, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR)

            img_FDA = dePhase1(img)

            img_FDA = img_FDA[:w, :h]
            if np.random.random() > 0.5: img_FDA = np.fliplr(img_FDA)  # 沿水平轴翻转数组
            if np.random.random() > 0.5: img_FDA = np.flipud(img_FDA)  # 沿水平轴翻转数组

            return img_FDA

        def addNoise(img0):
            # 2.高斯模糊
            img_FDA_guassian = cv2.GaussianBlur(img0, (13, 13), 0)
            # 3.添加点噪声
            noise_map = np.random.uniform(-5, 5, img_FDA_guassian.shape)
            img_FDA_guassian = img_FDA_guassian + noise_map
            return np.clip(img_FDA_guassian, 0., 255.)

        def deVessel(img, noise):
            img = np.expand_dims(img, axis=2)  # (512, 512) -> (512, 512, 1)
            noise = np.expand_dims(noise, axis=2)  # (512, 512) -> (512, 512, 1)
            img = img.transpose((2, 0, 1))  # 通过转置操作，改变维度顺序
            noise = noise.transpose((2, 0, 1))

            # 傅里叶变换 # get fft of both source and target
            fft_trg_np1 = np.fft.fft2(img, axes=(-2, -1))
            fft_trg_np2 = np.fft.fft2(noise, axes=(-2, -1))

            def low_freq_mutate_np2(amp_src1, amp_src2):  # L=0~255
                # amp_src, amp_trg: (1, 512, 512) <class 'numpy.ndarray'>
                a_src1 = np.fft.fftshift(amp_src1, axes=(-2, -1))  # np.zeros_like(amp_trg)
                a_src2 = np.fft.fftshift(amp_src2, axes=(-2, -1))
                # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。
                _, h, w = a_src1.shape  # h=512 w=512
                for i in range(h):
                    for j in range(w):
                        k = ((i - h / 2) ** 2 + (j - w / 2) ** 2) ** 0.5
                        # if k>10 and k<90 and not (k>50 and k<55): #(10,100)
                        if k > 10 and k < 100:  # (10,90)
                            # a_src1[0,i,j]=0
                            a_src1[0, i, j] = a_src2[0, i, j]

                return np.fft.ifftshift(a_src1, axes=(-2, -1))

            # extract amplitude and phase of both ffts
            amp_trg1, pha_trg1 = np.abs(fft_trg_np1), np.angle(fft_trg_np1)
            amp_trg2, pha_trg2 = np.abs(fft_trg_np2), np.angle(fft_trg_np2)

            # mutate the amplitude part of source with target
            amp_src_ = low_freq_mutate_np2(amp_trg1, amp_trg2)
            pha_src_ = low_freq_mutate_np2(pha_trg1, pha_trg2)

            # mutated fft of source
            fft_src_ = amp_src_ * np.exp(1j * pha_src_)

            # 逆傅里叶变换 # get the mutated image
            src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
            src_in_trg = np.real(src_in_trg)  # 从复数数组中提取实部
            # print("src_in_trg",src_in_trg.shape,src_in_trg)

            img_FDA = np.clip(src_in_trg, 0, 255.)  # 应该是限制像素的最小值为0、最大值为255
            img_FDA = np.squeeze(img_FDA, axis=0)  # (1, 512, 512) -> (512, 512)

            return img_FDA

        def deVessel2(img):
            ##########################    1.计算noise   ##########################
            img = np.expand_dims(img, axis=2)  # (512, 512) -> (512, 512, 1)
            img = img.transpose((2, 0, 1))  # 通过转置操作，改变维度顺序

            fft_trg_np = np.fft.fft2(img, axes=(-2, -1))  # 傅里叶变换

            # extract amplitude and phase of both ffts
            amp, pha = np.abs(fft_trg_np), np.angle(fft_trg_np)

            # mutated fft of source
            fft_0 = amp * np.exp(1j * np.zeros_like(pha))

            # 逆傅里叶变换 # get the mutated image
            src_0 = np.fft.ifft2(fft_0, axes=(-2, -1))
            src_0 = np.real(src_0)  # 从复数数组中提取实部。

            noise = np.clip(src_0, 0, 255.)  # 限制像素的最小值为0、最大值为255
            ##########################    2.计算noise   ##########################
            # 傅里叶变换 # get fft of both source and target
            fft_trg_np2 = np.fft.fft2(noise, axes=(-2, -1))

            def low_freq_mutate_np2(amp_src1, amp_src2):  # L=0~255
                a_src1 = np.fft.fftshift(amp_src1, axes=(-2, -1))  # np.zeros_like(amp_trg)
                a_src2 = np.fft.fftshift(amp_src2, axes=(-2, -1))
                # 频域中心化: 将频谱的零频率分量（DC分量）移动到频谱的中心位置。
                _, h, w = a_src1.shape  # h=512 w=512
                for i in range(h):
                    for j in range(w):
                        k = ((i - h / 2) ** 2 + (j - w / 2) ** 2) ** 0.5
                        if k > 10 and k < 100:
                            a_src1[0, i, j] = a_src2[0, i, j]
                return np.fft.ifftshift(a_src1, axes=(-2, -1))

            # extract amplitude and phase of both ffts
            amp_trg2, pha_trg2 = np.abs(fft_trg_np2), np.angle(fft_trg_np2)

            # mutate the amplitude part of source with target
            amp_src_ = low_freq_mutate_np2(amp, amp_trg2)
            pha_src_ = low_freq_mutate_np2(pha, pha_trg2)

            # mutated fft of source
            fft_src_ = amp_src_ * np.exp(1j * pha_src_)

            # 逆傅里叶变换 # get the mutated image
            src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
            src_in_trg = np.real(src_in_trg)  # 从复数数组中提取实部
            # print("src_in_trg",src_in_trg.shape,src_in_trg)

            img_FDA = np.clip(src_in_trg, 0, 255.)  # 应该是限制像素的最小值为0、最大值为255
            img_FDA = np.squeeze(img_FDA, axis=0)  # (1, 512, 512) -> (512, 512)

            return img_FDA

        if config.dePhase == 1:
            background_array = dePhase1(background_array, False)
        elif config.dePhase == 2:
            background_array = dePhase2(background_array)
        elif config.dePhase == 3:  # 经过分析、我不认为该方式会提升效果
            background_array = dePhase1(background_array, True)
            # background_array=dePhase1(background_array,False)
            # background_array=addNoise(background_array)
        elif config.dePhase == 4:
            background_array = deVessel2(img)
            # noiseImg = dePhase1(background_array,False)
            # background_array = deVessel(img,noiseImg)
        elif not config.dePhase == 0:  # dePhase=0表示程序执行前已经处理好了背景图
            print("The dePhase parameter in the configuration file is invalid!(配置文件中的dePhase参数不合法!)")
            exit(0)

        im_src = np.asarray(img, np.float32)  # 血管 <PIL.Image.Image> -> <numpy.ndarray> #转换为NumPy，并且指定类型
        background_array = background_array / 255.
        im_src = im_src / 255.
        # 血管图后面似乎可以尝试进行高斯模糊处理
        # print('background_array:',background_array)
        synthetic_img = background_array * im_src
        # print('synthetic_Image:', synthetic_Image,synthetic_Image.shape)
        # exit(0)
        synthetic_Image = Image.fromarray((255. * synthetic_img).astype('uint8')).convert('L')
        org_img_size = img.size

        if False:  # 保存Image对象为PNG格式的图片到本地
            print(img_name)
            synthetic_Image.save('output_image_' + img_name + '.png')
            exit(0)

        def getPerturbationImg(background_array0, im_src0):
            random_float = random.uniform(1, 4) # 生成一个1到4之间的随机小数
            im_src0 = im_src0 ** random_float
            synthetic_img0 = background_array0 * im_src0
            return Image.fromarray((255. * synthetic_img0).astype('uint8')).convert('L')

        synthetic_Image_Perturbation = getPerturbationImg(background_array, im_src)

        return synthetic_Image, anno_mask, org_img_size, synthetic_Image_Perturbation
        # img_FDA_Image, 数据
        # anno_mask,     标签
        # org_img_size   维度

    def load_frame_fakevessel_gaussian_intensity(self, img_name, background_name):  # 这个函数似乎没有被调用
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)

        background_array = np.array(background_img)
        gt_arrray = np.array(anno_mask)
        im_src = np.asarray(img, np.float32)
        im_src[gt_arrray[0, :, :] == 1] = 220  # change the src intensity
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src, axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.02)

        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA, axis=0)
        img_FDA_guassian = cv2.GaussianBlur(img_FDA, (13, 13), 0)

        img_FDA_Image = Image.fromarray((img_FDA_guassian).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_fakevessel_elastic(self, img_name, background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)
        gt_array = np.array(anno_mask)  # tensor tp array
        gt_mask = np.squeeze(gt_array, axis=0) * 255
        background_array = np.array(background_img)
        im_src = np.asarray(img, np.float32)
        im_trg = np.asarray(background_array, np.float32)
        im_src = np.expand_dims(im_src, axis=2)
        im_trg = np.expand_dims(im_trg, axis=2)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.02)

        img_FDA = np.clip(src_in_trg, 0, 255.)
        img_FDA = np.squeeze(img_FDA, axis=0)
        img_FDA_Image = Image.fromarray((img_FDA).astype('uint8')).convert('L')
        gt_Image = Image.fromarray((gt_mask).astype('uint8')).convert('L')
        image_deformed, mask_deformed = elastic_transform_PIL(img_FDA_Image, gt_Image, img_FDA.shape[1] * 2,
                                                              img_FDA.shape[1] * 0.2, img_FDA.shape[1] * 0.1)
        mask_deformed[mask_deformed == 0] = 0
        mask_deformed[mask_deformed == 255] = 1
        mask_deformed = torch.from_numpy(mask_deformed).float().unsqueeze(0)
        img_deform_Image = Image.fromarray((image_deformed).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_deform_Image, mask_deformed, org_img_size

    def load_frame_fakevessel_cutvessel(self, img_name, background_name):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)
        background_img = self.read_background(background_name)
        im_array = np.array(img)
        gt_arrray = np.array(anno_mask)

        background_array = np.array(background_img)
        img_FDA_r = np.where(gt_arrray[0, :, :] > 0, im_array, background_array)
        img_FDA_Image = Image.fromarray((img_FDA_r).astype('uint8')).convert('L')

        org_img_size = img.size

        return img_FDA_Image, anno_mask, org_img_size

    def load_frame_unsupervised(self, img_name):  # 加载一张真实的造影图，无监督的训练
        img = self.read_img(img_name)  # 加载一张真实的造影图
        org_img_size = img.size  # 造影图的尺寸
        return img, org_img_size  # 造影图，尺寸

    def load_frame_supervised(self, img_name, idx_background):
        img = self.read_img(img_name)
        anno_mask = self.read_mask(img_name)

        org_img_size = img.size

        return img, anno_mask, org_img_size

    def read_mask(self, img_name):
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name)).convert('L'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_mask_3D(self, img_name):
        mask = np.array(Image.open(os.path.join(self.ann_path_3D, img_name)).convert('L'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_ignore_mask(self, img_name):
        mask = np.array(Image.open(os.path.join(self.ignore_path, img_name) + '.png'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_boundary(self, img_name):
        mask = np.array(Image.open(os.path.join(self.bd_path, img_name) + '.png'))
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def read_img(self, img_name):
        # maybe png
        return Image.open(os.path.join(self.img_path, img_name)).convert('L')

    def read_img_3D(self, img_name):
        # maybe png
        return Image.open(os.path.join(self.img_path_3D, img_name)).convert('L')

    def read_background(self, img_name):
        return Image.open(os.path.join(self.background_path, img_name)).convert('L')

    def read_background_3D(self, img_name):
        return Image.open(os.path.join(self.background_path_3D, img_name)).convert('L')

    def load_metadata(self):
        if self.split == 'train':
            meta_file = os.path.join(self.datapath, 'split', 'train.txt')
        elif self.split == 'val' or self.split == 'test':
            meta_file = os.path.join(self.datapath, 'split', 'test.txt')
        else:
            raise RuntimeError('Undefined split ', self.split)

        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    def load_metadata_supervised(self):  # get all fake vessels file names written in train_fakevessel.txt
        # train_fakevessel.txt
        if self.split == 'train':
            meta_file = os.path.join(self.datapath, 'split', 'train_fakevessel.txt')

        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    def load_metadata_background(self):  # get all real img file names written in train_backvessel.txt
        # train_backvessel.txt
        if self.split == 'train':
            meta_file = os.path.join(self.datapath, 'split', 'train_backvessel.txt')
        print("unsupervised_metafile:", meta_file)
        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    # load_metadata_testsupervised
    def load_metadata_testsupervised(self):
        # test_img.txt
        if self.split == 'test' or 'val':
            meta_file = os.path.join(self.datapath, 'split', 'test_img.txt')
        record_fd = open(meta_file, 'r')
        records = record_fd.readlines()

        img_metaname = [line.strip() for line in records]
        return img_metaname

    def get_params(self, img, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th - 1)
        j = random.randint(0, w - tw - 1)
        return i, j, th, tw

    def get_params_fore(self, img, background_mask, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif isinstance(img, torch.Tensor) and img.dim() > 2:
                return img.shape[-2:][::-1]
            else:
                raise TypeError('Unexpected type {}'.format(type(img)))

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        B_i = random.randint(0, h - th - 1)
        B_j = random.randint(0, w - tw - 1)
        while (True):
            B_i = random.randint(0, h - th - 1)
            B_j = random.randint(0, w - tw - 1)
            background_crop = F.crop(background_mask, B_i, B_j, th, tw)
            sum_mask = torch.sum(background_crop)
            if sum_mask > 0:
                break
        return B_i, B_j, th, tw