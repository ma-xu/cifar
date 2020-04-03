#!/bin/bash

python3 cifar3.py --netName=ass6_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=ass7_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=ass8_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar3.py --netName=ass9_resnet18 --bs=128 --cifar=100

sleep 2m
sudo poweroff


