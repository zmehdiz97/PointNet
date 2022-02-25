from typing import OrderedDict
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pointnet import PointCloudData, PointMLP, \
PointNetBasic, PointNetFull, ShufflePoints, ToTensor

def test_transforms():
    return transforms.Compose([ShufflePoints(),ToTensor()])

def test(model, device, test_loader):

    correct = total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            if args.model == 'PointNetFull':
                outputs, __ = model(inputs.transpose(1,2))
            else:
                outputs = model(inputs.transpose(1,2))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100. * correct / total
    print('Test accuracy: %.1f %%' %(val_acc))


if __name__ == '__main__':

    # PARSER TO ADD OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["PointMLP", "PointNetBasic", "PointNetFull"], default="PointNetFull")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    print(f'Using device {device} \n')

    t0 = time.time()

    test_ds = PointCloudData("ModelNet40_PLY", folder='test', 
                            transform=test_transforms())

    inv_classes = {i: cat for cat, i in test_ds.classes.items()}
    print(f"Classes: {inv_classes} \n")
    print(f"Test dataset size: {len(test_ds)} \n")
    print(f"Number of classes: {len(test_ds.classes)} \n")
    print(f"Sample pointcloud shape: {test_ds[0]['pointcloud'].size()} \n")

    test_loader = DataLoader(dataset=test_ds, batch_size=32)
    if args.model == 'PointMLP':
        model = PointMLP()
    elif args.model == 'PointNetBasic':
        model = PointNetBasic()
    else:
        model = PointNetFull()

    print(f'Loading wights from checkpoint {args.checkpoint}')
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model.to(device);
    print('Testing ..')
    test(model, device, test_loader)
    
    print(f"Total time for testing : {time.time()-t0} \n")


