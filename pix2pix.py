import datetime
import os
import sys
import time

import cv2
import numpy as np
import torch

import torchvision.transforms as transforms
from torch.autograd import Variable

from Frames_dataset import FramesDataset
from models import *
from opts import parse_opts

opt = parse_opts()
print(opt)

os.makedirs("images_generate/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("save_models/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("result/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# 损失函数
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

lambda_pixel = 100

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# 初始化生成器和判别器
generator = GeneratorUNet()
discriminator = Discriminator()

# 使用gpu
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # 导入训练好的模型
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # 初始化权重
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 图像变换
transform=transforms.Compose([
                               transforms.Resize(opt.img_size),   #把图像的短边统一为image_size，另外一边做同样倍速缩放，不一定为image_szie
                               transforms.ToTensor(),
                           ])



# 创建数据迭代器
dataset = FramesDataset(opt,dataset='alice',transform=transform)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)


# Tensor 类型
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs[1].type(Tensor))
    real_B = Variable(imgs[0].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    cv2.imwrite("images_generate/%s/%s.png" % (opt.dataset_name, batches_done),255*img_sample[0].squeeze(0).cpu().swapaxes(0,2).swapaxes(0,1).numpy())


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # 输入 tensor shape[512,512]
        real_A = Variable(batch[1].type(Tensor))
        real_B = Variable(batch[0].type(Tensor))

        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # 总损失
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # 打印log 这段代码很神奇！！
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # 如果到达一定时间就保存图片
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 保存模型参数
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
