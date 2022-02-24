
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 12/01/2021
#

from typing import OrderedDict
import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F

# Import functions to read and write ply files
from ply import write_ply, read_ply

class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)

class RandomSymmetry(object):
    """ Apply a random symmetry transformation on the data

    Parameters
    ----------
    axis: Tuple[bool,bool,bool], optional
        axis along which the symmetry is applied
    """

    def __init__(self, axis=[False, False, False]):
        self.axis = axis

    def __call__(self, data):

        for i, ax in enumerate(self.axis):
            if ax:
                if torch.rand(1) < 0.5:
                    c_max = np.max(data[:, i])
                    data[:, i] = c_max - data[:, i]
        return data

    def __repr__(self):
        return "Random symmetry of axes: x={}, y={}, z={}".format(*self.axis)

class RandomScaleAnisotropic:
    r""" Scales node positions by a randomly sampled factor ``s1, s2, s3`` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \left[
        \begin{array}{ccc}
            s1 & 0 & 0 \\
            0 & s2 & 0 \\
            0 & 0 & s3 \\
        \end{array}
        \right]


    for three-dimensional positions.

    Parameters
    -----------
    scales:
        scaling factor interval, e.g. ``(a, b)``, then scale \
        is randomly sampled from the range \
        ``a <=  b``. \
    """

    def __init__(self, scales=None, anisotropic=True):
        assert len(scales) == 2
        assert scales[0] <= scales[1]
        self.scales = scales

    def __call__(self, data):
        scale = self.scales[0] + np.random.rand(3) * (self.scales[1] - self.scales[0])
        data = data * scale
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.scales)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ShufflePoints(),ToTensor()])

def custom_transforms():
    T = transforms.Compose([
        RandomRotation_z(),
        RandomNoise(),
        ShufflePoints(),
        RandomScaleAnisotropic(scales=(0.8, 1.2)),
        RandomSymmetry(axis = [True, True, True]),
        ToTensor()])
    return T


class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    sample = {}
                    sample['ply_path'] = new_dir+"/"+file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]['ply_path']
        category = self.files[idx]['category']
        data = read_ply(ply_path)
        pointcloud = self.transforms(np.vstack((data['x'], data['y'], data['z'])).T)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}



class PointMLP(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(3072,512)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(512, 256)),
            ('bn2', nn.BatchNorm1d(256)),
            ('relu2', nn.ReLU()),
            ('Dropout', nn.Dropout(0.3)),
            ('linear3', nn.Linear(256, classes)),
            ('bn3', nn.BatchNorm1d(classes)),
            ('softmax', nn.LogSoftmax()),

        ]))


    def forward(self, input):
        return self.model(input)



class PointNetBasic(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('MLP1', nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)),
            ('bn1', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),

            ('MLP2', nn.Conv1d(64, 64, 1)),
            ('bn2', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),

            ('MLP3', nn.Conv1d(64, 64, 1)),
            ('bn3', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),

            ('MLP4', nn.Conv1d(64, 128, 1)),
            ('bn4', nn.BatchNorm1d(128)),
            ('relu', nn.ReLU()),

            ('MLP5', nn.Conv1d(128, 1024, 1)),
            ('bn5', nn.BatchNorm1d(1024)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool1d(1024)),
            ('flatten', nn.Flatten()),

            ('MLPglob1', nn.Linear(1024, 512)),
            ('bnglob1', nn.BatchNorm1d(512)),
            ('relu', nn.ReLU()),

            ('MLPglob2', nn.Linear(512, 256)),
            ('bnglob2', nn.BatchNorm1d(256)),
            ('Dropout', nn.Dropout(0.3)),
            ('relu', nn.ReLU()),

            ('MLPglob3', nn.Linear(256, classes)),
            ('softmax', nn.LogSoftmax()),

        ]))

    def forward(self, input):
        return self.model(input)
        
        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.model = nn.Sequential(OrderedDict([
            ('MLP1', nn.Conv1d(3, 64, 1)),
            ('bn1', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),

            ('MLP2', nn.Conv1d(64, 128, 1)),
            ('bn2', nn.BatchNorm1d(128)),
            ('relu', nn.ReLU()),

            ('MLP3', nn.Conv1d(128, 1024, 1)),
            ('bn3', nn.BatchNorm1d(1024)),
            ('relu', nn.ReLU()),

            ('maxpool', nn.MaxPool1d(1024)),
            ('flatten', nn.Flatten()),

            ('MLPglob1', nn.Linear(1024, 512)),
            ('bnglob1', nn.BatchNorm1d(512)),
            ('relu', nn.ReLU()),

            ('MLPglob2', nn.Linear(512, 256)),
            ('bnglob2', nn.BatchNorm1d(256)),
            ('relu', nn.ReLU()),

            ('MLPglob3', nn.Linear(256, k*k)),
        ]))
    def forward(self, input):
        return self.model(input).reshape(-1,self.k,self.k)


class PointNetFull(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.TN1 = Tnet(k=3)
        self.block1 = nn.Sequential(OrderedDict([

            ('MLP1', nn.Conv1d(3, 64, 1)),
            ('bn1', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),

            ('MLP2', nn.Conv1d(64, 64, 1)),
            ('bn2', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),
        ]))
        #self.TN2 = Tnet(k=64)
        self.block2 = nn.Sequential(OrderedDict([
            ('MLP3', nn.Conv1d(64, 64, 1)),
            ('bn3', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),

            ('MLP4', nn.Conv1d(64, 128, 1)),
            ('bn4', nn.BatchNorm1d(128)),
            ('relu', nn.ReLU()),

            ('MLP5', nn.Conv1d(128, 1024, 1)),
            ('bn5', nn.BatchNorm1d(1024)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool1d(1024)),
            ('flatten', nn.Flatten()),

            ('MLPglob1', nn.Linear(1024, 512)),
            ('bnglob1', nn.BatchNorm1d(512)),
            ('relu', nn.ReLU()),

            ('MLPglob2', nn.Linear(512, 256)),
            ('bnglob2', nn.BatchNorm1d(256)),
            ('Dropout', nn.Dropout(0.3)),
            ('relu', nn.ReLU()),

            ('MLPglob3', nn.Linear(256, classes)),
            ('softmax', nn.LogSoftmax()),

        ]))

    def forward(self, input):
        tran1 = self.TN1(input)
        #print(input.shape, tran1.shape)
        x = input
        x = torch.transpose(torch.matmul(torch.transpose(x, 1, 2), tran1), 1, 2)
        x = self.block1(x)
        #print(x.shape)
        #tran2 = self.TN2(x)
        #x = torch.transpose(torch.matmul(torch.transpose(x, 1, 2), tran2), 1, 2)
        x = self.block2(x)
        return x, tran1



def basic_loss(outputs, labels):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)




def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    best_acc = 0
    for epoch in range(epochs): 
        avg_loss = 0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            #outputs = model(inputs.transpose(1,2))
            outputs, m3x3 = model(inputs.transpose(1,2))
            #loss = basic_loss(outputs, labels)
            loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
            avg_loss += loss

        model.eval()
        correct = total = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    #outputs = model(inputs.transpose(1,2))
                    outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, avg_loss / i, val_acc))
            if val_acc > best_acc:
                torch.save(model.state_dict(), 'PointNetFull.pth')
        scheduler.step()


if __name__ == '__main__':
    
    t0 = time.time()
    transform = custom_transforms()
    train_ds = PointCloudData("../data/ModelNet40_PLY", transform = transform)
    test_ds = PointCloudData("../data/ModelNet40_PLY", folder='test')

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    #model = PointMLP()
    #model = PointNetBasic()
    model = PointNetFull()
    print(str(model))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    model.to(device);

    train(model, device, train_loader, test_loader, epochs = 250)
    
    print("Total time for training : ", time.time()-t0)


