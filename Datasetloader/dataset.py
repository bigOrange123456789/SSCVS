r""" Dataloader builder """
from torch.utils.data import DataLoader
from Datasetloader.XCAD_liot import DatasetXCAD_aug
from Datasetloader.DRIVE_LIOT import DatasetDRIVE_aug
from Datasetloader.STARE_LIOT import DatasetSTARE_aug
from Datasetloader.Cracktree import DatasetCrack_aug


class CSDataset:

    @classmethod
    def initialize(cls, datapath):
        cls.datasets = {
            'XCAD_LIOT': DatasetXCAD_aug, # 用于处理XCAD数据集的对象
            'DRIVE_LIOT':DatasetDRIVE_aug,
            'STARE_LIOT': DatasetSTARE_aug,
            'Cracktree_LIOT':DatasetCrack_aug
        }
        cls.datapath = datapath # "./Data/XCAD"

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split, img_mode, img_size, supervised):
        # Force randomness during training for diverse episode combinations 在训练过错中强制随机划分(训练/测试集)
        # Freeze randomness during testing for reproducibility 为了可再现性测试、所以冻结随机性
        shuffle = split == 'train' # shuffle的本意是洗牌  # 训练集洗牌、验证集不洗牌
        nworker = nworker #if split == 'trn' else 0  #nworker的值是8,这里不知道是什么含义

        if split == 'train': # 训练集
            dataset = cls.datasets[benchmark](benchmark,            # XCAD_LIOT
                                              datapath=cls.datapath,# ./Data/XCAD
                                              split=split,          # train
                                              img_mode=img_mode,    # crop (译为裁减)
                                              img_size=img_size,    # 256
                                              supervised=supervised) # unsupervised/supervised
        else:  # split=val   # 验证集
            dataset = cls.datasets[benchmark](benchmark,            # XCAD_LIOT
                                              datapath=cls.datapath,# ./Data/XCAD
                                              split=split,          # val
                                              img_mode='same',
                                              img_size=None,
                                              supervised=supervised) # supervised

        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader,dataset
