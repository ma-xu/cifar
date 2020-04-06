#!/bin/bash

python3 cifar4.py --netName=old_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ac_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=double_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass1_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass2_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass3_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass4_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass5_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass6_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass7_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass8_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=ass9_resnet18 --bs=128 --cifar=100

sleep 1m
python3 cifar4.py --netName=triple_resnet18 --bs=128 --cifar=100


sleep 2m
sudo poweroff


