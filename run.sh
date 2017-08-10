#!/bin/bash

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
exec 2> .stderr
ROOT="features/cifar-10-matlab-batches"

echo "Experiments with 1000 labels and 300 anchors:"
printf "One hot encoding:"
python main.py --path $ROOT --labels 1000 --anchors 300 --code onehot
printf "LSH:"
python main.py --path $ROOT --labels 1000 --anchors 300 --code lsh --nbits 48
printf "Topline:"
python main.py --path $ROOT --labels 1000 --anchors 300
echo ""

echo "Experiments with 5000 labels and 1000 anchors:"
printf "One hot encoding:"
python main.py --path $ROOT --labels 5000 --anchors 1000 --code onehot
printf "LSH:"
python main.py --path $ROOT --labels 5000 --anchors 1000 --code lsh --nbits 64
printf "Topline:"
python main.py --path $ROOT --labels 5000 --anchors 1000
echo ""

echo "Experiment with 59000 labels and 1000 anchors: "
python main.py --path $ROOT --labels 59000 --anchors 1000
