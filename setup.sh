#!/bin/bash

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -e
mkdir -p features/cifar-10-matlab-batches
cd features/cifar-10-matlab-batches
wget http://pascal.inrialpes.fr/data2/asablayr/suphash/cifar10_gist.npy
wget http://pascal.inrialpes.fr/data2/asablayr/suphash/cifar10_label.npy
