#!/bin/bash
python3 cifar2.py --netName=old_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=ac_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=double_resnet50 --bs=128 --cifar=100

sleep 1m
python3 cifar2.py --netName=triple_resnet50 --bs=128 --cifar=100
