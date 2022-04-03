# unet网络分割左心室影像
使用unet训练网络，每个epoch结束后在验证集上进行评估，分别计算pixelAccuracy，IOU等指标，并将训练结果保存到log.txt文件夹中
使用softmax作为损失函数
~~dice系数还没有添加进去~~
iou = dice/(2-dice),dice = 2iou/(1+iou)