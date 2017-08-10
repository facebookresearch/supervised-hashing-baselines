#!/bin/bash

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
DIR="cifar-10-batches-mat"
if [ ! -d $DIR ]; then
  if [ ! -f cifar-10-matlab.tar.gz ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
  fi
  tar -xzvf cifar-10-matlab.tar.gz
fi

# Computing gists take ~1h on 8 cores
FILES=( data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch )
for file in "${FILES[@]}"
do
  if [ ! -f "$DIR/${file}_gist.mat" ]; then
    ./cifar10_gist.m "$DIR/${file}.mat" &
  fi
done
wait
