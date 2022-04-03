import torch
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk
times = 2
# '/public/home/heyinte/guoziyu/training'
# '/public/home/heyinte/guoziyu/validate'
class Heart_Loader(Dataset):
    def __init__(self):
        '''
        根据标注文件去取图片
        '''
        self.imagePath = 'E:/training/training'
        # self.imagePath = '/mnt/data/guoziyu/training'
        self.totensor = torchvision.transforms.ToTensor()
        self.resizer = torchvision.transforms.Resize((256,256))

    def __len__(self):
        # return len(os.listdir(self.imagePath))*times
        return 10

    def __getitem__(self, item):
        item = item+1
        sum = len(os.listdir(self.imagePath))
        if(item <= sum):
            patient = 'patient' + format(item, '04d')
            image_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ED.mhd'
            label_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ED_gt.mhd'
        else:
            patient = 'patient' + format(item-sum, '04d')
            image_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ES.mhd'
            label_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ES_gt.mhd'
        image_rawfile = sitk.ReadImage(image_path)
        label_rawfile = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(image_rawfile)
        label = sitk.GetArrayFromImage(label_rawfile)
        # width ,height = label[0].shape
        # for i in range(0,width):
        #     for j in range(0,height):
        #         print(label[0][i][j])

        image = torchvision.transforms.ToPILImage()(image[0])
        label = torchvision.transforms.ToPILImage()(label[0])
        image = self.resizer(image)
        label = self.resizer(label)
        image = self.totensor(image)
        label = self.totensor(label)
        # print(image.shape)
        label = label*255
        return image, label
class Valid_Loader(Dataset):
    def __init__(self):
        '''
        根据标注文件去取图片
        '''
        self.imagePath = 'E:/training/validate'
        # self.imagePath = '/mnt/data/guoziyu/validate'
        self.totensor = torchvision.transforms.ToTensor()
        self.resizer = torchvision.transforms.Resize((256,256))

    def __len__(self):
        return len(os.listdir(self.imagePath))*times

    def __getitem__(self, item):
        item = item+1
        sum = len(os.listdir(self.imagePath))
        if(item <= sum):
            patient = 'patient' + format(item+400, '04d')
            image_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ED.mhd'
            label_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ED_gt.mhd'
        else:
            patient = 'patient' + format(item-sum+400, '04d')
            image_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ES.mhd'
            label_path = self.imagePath + '/' + patient + '/' + patient + '_2CH_ES_gt.mhd'
        image_rawfile = sitk.ReadImage(image_path)
        label_rawfile = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(image_rawfile)
        label = sitk.GetArrayFromImage(label_rawfile)
        # width ,height = label[0].shape
        # for i in range(0,width):
        #     for j in range(0,height):
        #         print(label[0][i][j])

        image = torchvision.transforms.ToPILImage()(image[0])
        label = torchvision.transforms.ToPILImage()(label[0])
        image = self.resizer(image)
        label = self.resizer(label)
        image = self.totensor(image)
        label = self.totensor(label)
        # print(image.shape)
        label = label*255
        return image, label
