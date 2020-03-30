#!/bin/bash
python3 cifar2.py --netName=old_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=ass1_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=ass2_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=ass3_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=ass4_resnet50 --bs=128 --cifar=100

sudo poweroff

