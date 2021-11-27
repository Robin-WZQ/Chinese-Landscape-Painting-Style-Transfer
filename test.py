import argparse

import cv2
import numpy
import PIL.Image
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from models import *
from tools.canny import processing
from tools.picture2texture import estimate


def sample_images(generator,Tensor,imgs):
    """
    save the processed pictures

    Args:
        generator: trained model
        Tensor: tensor format
        imgs: real picture
    
    Author: Zhongqi Wang
    """
    real_A = Variable(imgs.type(Tensor))
    real_A = real_A.unsqueeze(0)
    fake_B = generator(real_A)
    cv2.imwrite("generate.png" ,255*fake_B[0].squeeze(0).cpu().swapaxes(0,2).swapaxes(0,1).numpy())

def process(opt,file_path):
    """
    get the HED edge-painting

    Args:
        opt: opt file
        file_path: the file path U want to process
    
    Author: Zhongqi Wang
    """
    arguments_strOut = "HED.jpg"
    src = cv2.imread(file_path, 0)
    src = cv2.resize(src, (opt.img_width,opt.img_height))
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    a = PIL.Image.fromarray(src_RGB)
    b = numpy.array(a)[:, :]
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(b.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenOutput = estimate(tenInput)
    PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

def main(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    opt = parser.parse_args()

    transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])

    cuda = True if torch.cuda.is_available() else False
    generator = GeneratorUNet()
    if cuda:
        generator = generator.cuda() #使用gpu
    generator.load_state_dict(torch.load("generator_45_canny.pth"))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    
    process(opt,path) #处理为HED边缘图像
    img = processing(path) #处理为canny边缘图像
    cv2.imwrite("canny.jpg",img)
    pic1 = cv2.imread("HED.jpg")
    pic1 = cv2.resize(pic1, (opt.img_width,opt.img_height))
    pic2 = cv2.imread("canny.jpg")
    pic2 = cv2.resize(pic2, (opt.img_width,opt.img_height))
    train_data = pic1+pic2
    cv2.imwrite("canny&HED.jpg",train_data) #得到二者叠加
    frame = cv2.resize(train_data,(opt.img_width,opt.img_height))
    frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    frame = transform(frame)
    sample_images(generator,Tensor,frame) #输入pix2pix模型求解


if __name__ == "__main__":
    path = "test_pic/6.jpg" # 要处理的图片
    main(path)
