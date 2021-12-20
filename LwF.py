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

class HisNet(nn.Module):
    def __init__(self):
        super(HisNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 10, 3),
            nn.Flatten(),
            # nn.Linear(128 * 3 * 3, 512),
            nn.Linear(10 * 1 * 1, 5)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


net = HisNet()
net.load_state_dict(torch.load("model.pth"))
net.eval()


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


pathname = 'E:\dataset\data1'
train_label, test_label = mat2pic(pathname)

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


transform = transforms.Compose([
    transforms.CenterCrop(88),
    transforms.ToTensor()
])


train_dataloader = DataLoader(Mydata('E:\learningpytorch\Train1', train_label, transform), batch_size=50, drop_last=True, shuffle=True)
test_dataloader = DataLoader(Mydata('E:\learningpytorch\Test1', test_label, transform), batch_size=50, drop_last=True, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = HisNet().to(device)
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
# epochs = 100


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.long())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct

# train(train_dataloader, net, loss_fn, optimizer)
test(test_dataloader, net, loss_fn)

'''
t = 1
acc = 0
flag = 0
while True:
    print(f'Epoch {t}: \n--------------------')
    train(train_dataloader, net, loss_fn, optimizer)
    acc = test(test_dataloader, net, loss_fn)
    if acc > 93:
        flag += 1
    if flag > 5:
        break
    t += 1

print("DONE")
'''
