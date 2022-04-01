import SimpleITK as sitk
import torch.nn as nn
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import cv2
import numpy as np
import glob
from torchvision import models
def readraw(path):
    image_rawfile = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image_rawfile)
    image = np.squeeze(image[0, ...])
    cv2.imwrite(path.replace('mhd', "jpg"),image)
    image = cv2.resize(image, (256, 256))
    # image = torchvision.transforms.ToPILImage()(image)
    tensor = torchvision.transforms.ToTensor()
    image = tensor(image)
    # image = image.reshape(1, 1, 256, 256)
    image = image.unsqueeze(0)
    return image
import torch.nn.functional as F
from PIL import Image

from PIL import ImageEnhance
from unet_model import UNet as MyUnet
def test_heart(i):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net = ISBI_UNET(1, 1)
    net = MyUnet(1,1)
    # 将网络拷贝到deivce中
    # net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_class49.pth', map_location=device))
    # 测试模式
    net.eval()
    num = format(i,'04d')
    path = 'E:\\training\\validate\\patient'+num+'\patient'+num+'_2CH_ED.mhd'
    labelpath = 'E:\\training\\validate\patient'+num+'\patient'+num+'_2CH_ED_gt.mhd'
    image = readraw(path)
    label = readraw(labelpath)
    # image = image.to(device=device)
    image = image.to(dtype=torch.float32)
    pred = net(image)
    # pred = F.softmax(pred,dim=1)
    # pred = pred.squeeze(0)
    # pred = torch.max(pred, dim=0)[1]
    # pred = pred.unsqueeze(0)
    # pred = pred.to(dtype=torch.float32)
    pred = torch.sigmoid(pred)
    pred = pred +0.0025
    # print((pred*255).max())
    # print((pred*255).min())
    img_p = torchvision.transforms.ToPILImage()(pred[0])
    img_s = torchvision.transforms.ToPILImage()(image[0])
    img_l = torchvision.transforms.ToPILImage()(label[0])
    # print(pred.detach().numpy())
    # print(label.detach().numpy())
    # print(label)
    # plt.subplot(121), plt.imshow(img_p, cmap="gray"), plt.title('res')
    # plt.subplot(122), plt.imshow(img_l, cmap='gray'), plt.title('src')
    # plt.show()
    imageName = 'patient'+num+'_2CH_ED_gt.'
    img_p = imageEhance(img_p)
    img_p.save('E:\\training\\result-unet\\validate\\unet-'+imageName+'jpg')
    # img_l = imageEhance(img_l)
    # img_l.save('E:\\training\\result-unet\\testing\\'+imageName+'jpg')
def imageEhance(img):
    img = np.array(img)
    img[img == 0] = 0
    img[img == 1] = 85
    img[img == 2] = 170
    img[img == 3] = 255
    return torchvision.transforms.ToPILImage()(img)
def getcontours(label):
    image, contours, her = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    label[label != 1] = 0
    image, contours2, her = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours,contours2
def contour(i):
    num = format(i, '04d')
    path = 'E:\\training\\result-unet\\validate' + '\\unet-patient' + num + '_2CH_ES_gt.jpg'
    labelpath = 'E:\\training\\validate\\patient' + num + '\\patient' + num + '_2CH_ES_gt.mhd'
    label = readraw(labelpath)
    label = torchvision.transforms.ToPILImage()(label[0])
    label = np.array(label)
    contours,contours2 = getcontours(label)
    net = MyUnet(1, 1)

    image = cv2.imread(path)
    # print(image)
    # im = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    # label = cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
    cv2.drawContours(image,contours,-1,(0,0,255),2)
    cv2.drawContours(image, contours2, -1, (0, 0, 255), 2)
    # cv2.imwrite('res1.jpg',image)
    # cv2.imshow('ds',image)
    # cv2.waitKey(0)
    cv2.imwrite('E:\\training\\result-unet\\validate_circle\\'+num+'_2CH_ES_gt.jpg',image)

def makeraw(i):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MyUnet(1, 1)
    # 将网络拷贝到deivce中
    # net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_class1_4CH9.pth', map_location=device))
    # 测试模式
    net.eval()
    num = format(i, '04d')
    path = 'E:\\testing\\testing\\patient' + num + '\\patient' + num + '_4CH_ED.mhd'
    image = readraw(path)
    pred = net(image)
    pred = torch.sigmoid(pred)
    pred = pred + 0.0025
    img_p = torchvision.transforms.ToPILImage()(pred[0])
    image = np.array(img_p)
    # print(image.max())
    image[image == 0] = 0
    image[image == 1] = 1
    image[image == 2] = 2
    image[image >= 3] = 3
    # print(image.max())
    src = cv2.imread('E:\\testing\\testing\\patient' + num + '\\patient' + num + '_2CH_ED.jpg')
    height,width,channels = src.shape
    # print(src.shape)
    raw = cv2.resize(image,(width,height))
    # cv2.imwrite('test.jpg',raw)
    # raw.tofile('res.raw')
    # print(image.shape)
    # raw = raw.reshape(1,width,height)
    raw = raw.reshape(height,width,1)
    raw = sitk.GetImageFromArray(raw,isVector=True)
    # print(raw.GetSize())
    sitk.WriteImage(raw,'E:\\training\\result-unet\\submit\\patient'+ num + '_4CH_ED.mhd')
