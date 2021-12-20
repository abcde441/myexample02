import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models


def MultiClassCrossEntropy(logits, labels, T):
    outputs = torch.log_softmax(logits / T, dim=1)
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class HisNet(nn.Module):
    def __init__(self, classes, classes_map, args):
        self.init_lr = args.init_lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)]
        self.lr_dec_factor = 10

        self.pretrained = False
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.epsilon = 1e-16
        super(HisNet, self).__init__()
        self.model = models.resnet34(pretrained=self.pretrained)
        self.model.apply(kaiming_normal_init)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor)
        self.n_class = 0
        self.n_know = 0
        self.classes_map = classes_map

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def increment_class(self, new_classes):
        n = len(new_classes)
        print(f'new class:{n}')
        in_features = self.fc.in_features
        out_features = self.fc.out_featrues
        weight = self.fc.weight.data
