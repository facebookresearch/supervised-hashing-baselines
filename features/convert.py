# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from os.path import join

root = "cifar-10-batches-mat"

files = ["data_batch_%d_gist.mat" % i for i in range(1, 6)]
files.append("test_batch_gist.mat")

gist = np.vstack([np.loadtxt(join(root, f)) for f in files])
files = [f.replace("gist", "label") for f in files]

label = [np.reshape(np.loadtxt(join(root, f)), (-1, 1)) for f in files]
label = np.vstack(label)

np.save(join(root, "cifar10_gist.npy"), gist)
np.save(join(root, "cifar10_label.npy"), label)
