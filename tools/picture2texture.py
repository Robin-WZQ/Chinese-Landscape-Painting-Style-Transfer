#this file is modified by Zhongqi Wang from https://github.com/sniklaus/pytorch-hed

#!/usr/bin/env python

import getopt
import os
import sys

import cv2
import numpy
import PIL
import PIL.Image
import torch

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##########################################################

arguments_strModel = 'bsds500' # only 'bsds500' for now


for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.netVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.netCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items() })
	# end

	def forward(self, tenInput):
		tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
		tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
		tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

		tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

		tenVggOne = self.netVggOne(tenInput)
		tenVggTwo = self.netVggTwo(tenVggOne)
		tenVggThr = self.netVggThr(tenVggTwo)
		tenVggFou = self.netVggFou(tenVggThr)
		tenVggFiv = self.netVggFiv(tenVggFou)

		tenScoreOne = self.netScoreOne(tenVggOne)
		tenScoreTwo = self.netScoreTwo(tenVggTwo)
		tenScoreThr = self.netScoreThr(tenVggThr)
		tenScoreFou = self.netScoreFou(tenVggFou)
		tenScoreFiv = self.netScoreFiv(tenVggFiv)

		tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

		return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))
	# end
# end

netNetwork = None

##########################################################

def estimate(tenInput):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().to(device).eval()
	# end

	intWidth = tenInput.shape[2]
	intHeight = tenInput.shape[1]

	assert(intWidth == 512) # 调整为本次图像输入大小
	assert(intHeight == 512) # 调整为本次图像输入大小

	return netNetwork(tenInput.to(device).view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
	files_path = []
	input_path = "E:/junior/CV/Homework/term_project/generate/shashui/Landscape_GAN/pic2" #输入路径
	os.makedirs("processing_pic/" , exist_ok=True)
	names = ['Harvard','met-1','met-2','Princeton-1','Princeton-2','Smithsonian-1','Smithsonian-2','Smithsonian-3','Smithsonian-4','Smithsonian-5']
	for name in names:
		os.makedirs("processing_pic/%s" % name , exist_ok=True)
	output_path = "processing_pic" #输出路径


	for label in sorted(os.listdir(input_path)): #label：来源哪个数据集
		for fname in os.listdir(os.path.join(input_path, label)):
			files_path.append(os.path.join(input_path, label, fname)) #图片的文件名
	
	for file_path in files_path:
		arguments_strIn = file_path
		all = file_path.split("\\")
		label = all[-2] #获取数据集名，如met-1
		fname = all[-1] #获取文件名，如met_0.jpg
		arguments_strOut = os.path.join(output_path, label, fname)
		try:
			tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
			tenOutput = estimate(tenInput)
			PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
		except:
			'''若是灰度图需要特别处理！'''
			src = cv2.imread(file_path, 0)
			src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
			a = PIL.Image.fromarray(src_RGB)
			b = numpy.array(a)[:, :]
			tenInput = torch.FloatTensor(numpy.ascontiguousarray(b.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
			tenOutput = estimate(tenInput)
			PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
		
