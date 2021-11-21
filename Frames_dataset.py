import os

import cv2
from PIL import Image
from torch.utils.data import Dataset


class FramesDataset2(Dataset):
    """
    This is the dataset function to get all pictures and their texture map.\n

    Returns:
        picture,texture (numpy format) 

    Author: Zhongqi Wang
    """

    def __init__(self,opt,dataset,transform=None):
        self.picture_path = opt.output_path
        self.texture_path = opt.texture_path
        self.resize_height = opt.img_size
        self.resize_width = opt.img_size
        self.transform = transform

        self.fnames = []
        self.fnames2 = []
        if dataset == "alice":
            for label in sorted(os.listdir(self.picture_path)): #label：来源哪个数据集
                for fname in os.listdir(os.path.join(self.picture_path, label)):
                    self.fnames.append(os.path.join(self.picture_path, label, fname)) #文件名，还没到帧图片
            for label in sorted(os.listdir(self.texture_path)): #label：来源哪个数据集
                for fname in os.listdir(os.path.join(self.texture_path, label)):
                    self.fnames2.append(os.path.join(self.texture_path, label, fname)) #文件名，还没到帧图片


        elif dataset == 'my_own':
            for label in sorted(os.listdir(self.picture_path)): #label：来源哪个数据集
                for fname in os.listdir(os.path.join(self.picture_path, label)):
                    self.fnames.append(os.path.join(self.picture_path, label, fname)) #文件名，还没到帧图片
            for label in sorted(os.listdir(self.texture_path)): #label：来源哪个数据集
                for fname in os.listdir(os.path.join(self.texture_path, label)):
                    self.fnames2.append(os.path.join(self.texture_path, label, fname)) #文件名，还没到帧图片

        assert(len(self.fnames)==len(self.fnames2))

        print('Number of pictures: {:d}'.format(len(self.fnames))) #输出获取到图片的个数
        print('Number of textures: {:d}'.format(len(self.fnames2))) #输出获取到纹理的个数

        try:
            self.check_integrity()
        except IndexError:
            raise RuntimeError('Dataset not found or corrupted.' +' You need to download and preprocess it first.')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        # picture shape[16,3,112,112] -> [number of video clips,channels,width,height]

        picture,texture = self.load_frames(self.fnames[index],self.fnames2[index])
        if self.transform is None:
            raise ValueError("Transform is None! Please define it first.")

        return (picture,texture)

    def check_integrity(self):
        if not os.path.exists(self.picture_path):
            return False
        else:
            return True

    def load_frames(self, file_dir1,file_dir2):

        frame1 =cv2.imread(file_dir1)
        frame1 = Image.fromarray(frame1)
        frame1 = self.transform(frame1)

        frame2 =cv2.imread(file_dir2)
        frame2 = Image.fromarray(frame2)

        # # ==========================================================================
        # # 这里我将图片转成了灰度图，方便GAN网络计算，因为算力不够只能用小网络
        # # 可以不这样操作也ok
        # frame2 = frame2.convert('L')  # (option)
        # # ==========================================================================
        frame2 = self.transform(frame2)

        return frame1,frame2


