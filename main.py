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
from dataset import *
from losses import LovaszLossSoftmax
from unet_model import UNet as MyUnet
import torch.nn.functional as F
from metrics import *
from resnet import ResNet18
def train_HEART(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epochs=30, batch_size=1, lr=0.001):
    # 加载训练集
    train_dataset = Heart_Loader()
    valid_dataset = Valid_Loader()
    # checkpoint = torch.load('best_model_class2CH_290.pth')
    net = MyUnet(1,4)
    # net = ResNet18(BatchNorm=nn.BatchNorm2d, pretrained=False, output_stride=8)
    # net.load_state_dict(checkpoint)
    net = net.to(device)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=1,
                                               shuffle=True)
    # 定义RMSprop算法
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    # 定义Loss算法
    criterion = LovaszLossSoftmax()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    losses = []
    f = open('log.txt','w+')
    # 训练epochs次
    for epoch in range(epochs):
        print('开始第{}轮'.format(epoch + 1))
        f.write('epoch:{}\n'.format(epoch + 1))
        # 训练模式
        net.train()
        epoch_loss = 0.0
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device,dtype = torch.float32)
            label = label.to(device=device,dtype = torch.long)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # print(type(pred))
            # 计算loss
            loss = criterion(pred,label)
            # print((loss.item()))
            losses.append(float(loss.item()))
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                # torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        # print('bestloss:{}'.format(best_loss.item()))
        print('trainloss:{}'.format(sum(losses)/len(losses)))
        f.write('trainloss:{}\n'.format(sum(losses)/len(losses)))
        if((epoch +1) %50 == 0):
            torch.save(net.state_dict(), 'best_model_class2CH_{}.pth'.format(epoch))
        net.eval()
        mpalist = []
        mioulist = []
        palist = []
        ioulist = []
        cpalist = []
        for image, label in valid_loader:
            # 将数据拷贝到device中
            image = image.to(device=device,dtype = torch.float32)
            label = label.to(device=device,dtype = torch.long)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            # loss = criterion(pred,label)
            # print(pred.shape)
            # print(label.shape)
            # label = label.reshape(1,256,256)
            label = label[0]
            loss = F.cross_entropy(pred,label.long())
            pred = F.softmax(pred, dim=1)
            pred = pred[0]
            # pred = pred.reshape(4, 256, 256)
            pred = torch.max(pred, dim=0)[1]
            label = label[0]
            # label = label.reshape(256,256)
            # print(pred)
            # print(label)
            pred = pred.to(device='cpu',dtype = torch.float32)
            label = label.to(device='cpu',dtype = torch.float32)
            metric = SegmentationMetric(4)
            metric.addBatch(pred,label)
            pa = metric.pixelAccuracy()
            cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            IoU = metric.IntersectionOverUnion()
            mIoU = metric.meanIntersectionOverUnion()
            mpalist.append(float(mpa))
            mioulist.append(float(mIoU))
            palist.append(pa)
            ioulist.append(IoU)
            cpalist.append(cpa)
            # print(IoU)
            # print('mPA is : %f' % mpa)
            # print(mIoU)
            # print('accuracy:{}'.format(accuracy(pred,label)))
            # print('iou:{}'.format(iou(pred,label)))
            # print((loss.item()))
            losses.append(float(loss.item()))
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                # torch.save(net.state_dict(), 'best_model.pth')
        # print('bestloss:{}'.format(best_loss.item()))
        print('validloss:{}'.format(sum(losses)/len(losses)))
        f.write('validloss:{}\n'.format(sum(losses)/len(losses)))
        print('pixelAccuracy:{}'.format(sum(palist)/len(palist)))
        f.write('pixelAccuracy:{}\n'.format(sum(palist)/len(palist)))
        print('meanPixelAccuracy:{}'.format(sum(mpalist)/len(mpalist)))
        f.write('meanPixelAccuracy:{}\n'.format(sum(mpalist)/len(mpalist)))
        mIoU = sum(mioulist)/len(mioulist)
        print('mIoU:{}'.format(sum(mioulist)/len(mioulist)))
        f.write('mIoU:{}\n'.format(sum(mioulist)/len(mioulist)))
        print('Dice:{}'.format(2*mIoU/(1+mIoU)))
        f.write('Dice:{}\n'.format(2*mIoU/(1+mIoU)))
        data = np.array(ioulist)
        # print(data.shape)
        print('iou:{}'.format(np.average(data,axis=0)))
        f.write('iou:{}\n'.format(np.average(data,axis=0)))
        data = np.array(cpalist)
        print('classPixelAccuracy:{}'.format(np.average(data, axis=0)))
        f.write('classPixelAccuracy:{}\n'.format(np.average(data, axis=0)))
    f.close()





if __name__ == '__main__':
    train_HEART()

