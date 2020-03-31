#!/bin/bash
python3 cifar3.py --netName=old_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=ac_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=ass5_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=double_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=ass4_resnet18 --bs=128 --cifar=100


sudo poweroff


