import numpy as np
import scipy.io
import torchvision.transforms as transforms
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''
def mat2pic(pathname):
    filenames = os.listdir(pathname)
    step = 0
    train_label = []
    test_label = []
    for file in filenames:
        print('load dataset:' + str(file))
        mat = scipy.io.loadmat(pathname + '/' + file)
        train_img = np.transpose(np.uint8(mat['train_img']))
        test_img = np.transpose(np.uint8(mat['test_img']))
        np.save('./train_img1', train_img)
        np.save('./test_img1', test_img)
        train_img_dataset = np.load('./train_img1.npy')
        for i in range(train_img_dataset.shape[2]):
            img_tensor = train_img_dataset[:, :, i]
            img = transforms.ToPILImage()(img_tensor)
            img.save('./Train1/%s.%d.jpg' % (str(file.split('.')[0]), i))
        train_label1 = np.ones(train_img_dataset.shape[2], dtype=np.int16) * step
        test_img_dataset = np.load('./test_img1.npy')
        for j in range(test_img_dataset.shape[2]):
            img_tensor = test_img_dataset[:, :, j]
            img = transforms.ToPILImage()(img_tensor)
            img.save('./Test1/%s.%d.jpg' % (str(file.split('.')[0]), j))
        test_label1 = np.ones(test_img_dataset.shape[2], dtype=np.int16) * step
        train_label = np.hstack((train_label, train_label1))
        test_label = np.hstack((test_label, test_label1))
        step += 1
    return train_label, test_label
'''


class Mydata(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.image_path_list = os.listdir(self.root_dir)
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_index = self.image_path_list[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        img = img.resize((128, 128))
        label = self.label[index]

        if self.transform:
            img = self.transform(img)

        return img, label


'''
transform = transforms.Compose([
    transforms.CenterCrop(88),
    transforms.ToTensor()
])
'''

# train_dataloader = DataLoader(Mydata('E:\learningpytorch\Train1', train_label, transform), batch_size=50,
# drop_last=True, shuffle=True)
# test_dataloader = DataLoader(Mydata('E:\learningpytorch\Test1', test_label,
# transform), batch_size=50, drop_last=True, shuffle=True)
