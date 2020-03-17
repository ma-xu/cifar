#!/bin/bash
python3 cifar.py --netName=old_resnet50 --bs=512 --cifar=100

sleep 1m
python3 cifar.py --netName=ac_resnet50 --bs=512 --cifar=100

sleep 1m
python3 cifar.py --netName=double_resnet50 --bs=512 --cifar=100

sleep 1m
python3 cifar.py --netName=triple_resnet50 --bs=512 --cifar=100
