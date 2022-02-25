#!/bin/sh
#conda activate pytorch 
python pointnet.py --model PointMLP --aug default --epoch 30
python pointnet.py --model PointNetBasic --aug default --epoch 35
python pointnet.py --model PointNetFull --aug default --epoch 45
python pointnet.py --model PointNetFull --aug custom --epoch 50
