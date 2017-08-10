# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import argparse
import faiss
from os.path import join


def computemAP(q):
    assert type(q) == np.ndarray
    assert q.ndim == 2
    invvalues = np.divide(np.ones(q.shape[1]), np.ones(q.shape[1]).cumsum())

    map_ = 0
    prec_sum = q.cumsum(1)
    for i in range(prec_sum.shape[0]):
        idx_nonzero = np.nonzero(q[i])[0]
        if len(idx_nonzero) > 0:
            map_ += np.mean(prec_sum[i][idx_nonzero] * invvalues[idx_nonzero])

    return map_ / q.shape[0]


def getSigma(dataset, anchors):
    distances = scipy.spatial.distance.cdist(dataset, anchors)
    sigma = distances.min(1).mean()
    return sigma


def computeRBF(dataset, anchors, sigma):
    n = dataset.shape[0]
    features = np.zeros((n, anchors.shape[0]))
    distances = scipy.spatial.distance.cdist(dataset, anchors)
    features = np.exp((-1 / (2 * sigma**2)) * np.power(distances, 2))
    return features


def balancedSplit(X, y, seed, test_sz=1000):
    stratSplit = StratifiedShuffleSplit(
        y, 1, test_size=test_sz, random_state=seed
    )
    for train_idx, test_idx in stratSplit:
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        break
    return X_train, y_train, X_test, y_test


def getBalancedSample(y, seed, test_sz=1000):
    if y.shape[0] == test_sz:
        return np.arange(test_sz)
    else:
        stratSplit = StratifiedShuffleSplit(
            y, 1, test_size=test_sz, random_state=seed
        )
        for _, test_idx in stratSplit:
            idx = test_idx
            break
        return idx


def getmAP(clf, X_base, y_base, X_query, y_query, id_label, y_label):
    y_base, y_query = y_base[:, 0], y_query[:, 0]

    oneh = OneHotEncoder(sparse=False)
    y_label_1h = oneh.fit_transform(y_label)
    activations = clf.predict_proba(X_base)
    activations[id_label] = y_label_1h

    if args.code == "onehot":
        argmax = activations.argmax(axis=1).reshape((-1, 1))
        activations = oneh.fit_transform(argmax)

    if args.code == "lsh":
        index = faiss.IndexLSH(y_label_1h.shape[1], args.nbits, True, True)
    else:
        index = faiss.IndexFlatIP(y_label_1h.shape[1])

    index.train(activations.astype(np.float32))
    index.add(activations.astype(np.float32))

    queryAct = clf.predict_proba(X_query).astype(np.float32)

    _, idc = index.search(queryAct, y_base.shape[0])
    predictions = y_base[idc]
    results = np.equal(predictions, np.expand_dims(y_query, axis=1))

    return computemAP(results)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--code', choices=['lsh', 'onehot', 'none'], default='none')
parser.add_argument('--anchors', type=int, default=1000)
parser.add_argument('--labels', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--nbits', type=int)
parser.add_argument('--path', type=str, default="features")
args = parser.parse_args()

assert (args.code != "lsh" or args.nbits is not None)
np.random.seed(args.seed)

cifargist = np.load(join(args.path, "cifar10_gist.npy"))
cifarlabels = np.load(join(args.path, "cifar10_label.npy"))

X_train, y_train, X_test, y_test = balancedSplit(cifargist, cifarlabels, test_sz=1000, seed=args.seed)
nvalid = args.labels / 10

# Use 10% of labelled data for validation
id_label = getBalancedSample(y_train, seed=args.seed, test_sz=args.labels)
id_label_valid = id_label[-nvalid:]
id_label_train = id_label[:-nvalid]

anchors = X_train[id_label[np.random.choice(args.labels, args.anchors, False)]]
sigma = getSigma(X_train, anchors)
X_train = computeRBF(X_train, anchors, sigma)
X_test = computeRBF(X_test, anchors, sigma)

X_label, y_label = X_train[id_label], y_train[id_label]
mask_valid = np.array([(i in id_label_valid) for i in range(len(X_train))], dtype=bool)

# Get indices of labelled data in train \ valid
id_label_train_in_novalid = np.zeros((X_train.shape[0]))
id_label_train_in_novalid[~mask_valid] = np.arange(np.sum(~mask_valid))
id_label_train_in_novalid = id_label_train_in_novalid[id_label_train].astype(int)

# Cross valid regularization parameter
best_C, best_score = None, 0
for C in [2**k for k in range(-5, 13)]:
    clf = LogisticRegression(C=C, random_state=args.seed, n_jobs=1)
    clf.fit(X_train[id_label_train], y_train[id_label_train])
    train_score = clf.score(X_train[id_label_train], y_train[id_label_train])
    val_score = clf.score(X_train[id_label_valid], y_train[id_label_valid])
    val_map = getmAP(clf,
                     X_train[~mask_valid],
                     y_train[~mask_valid],
                     X_train[mask_valid],
                     y_train[mask_valid],
                     id_label_train_in_novalid,
                     y_train[id_label_train]
                     )

    if val_map >= best_score:
        best_C = C
        best_score = val_map

clf = LogisticRegression(C=best_C, random_state=args.seed, n_jobs=1)
clf.fit(X_label, y_label)

mAP = getmAP(clf,
             X_train,
             y_train,
             X_test,
             y_test,
             id_label,
             y_train[id_label]
             )

print(" %.3f mAP (C=%.2f) (%.11f)" % (mAP, best_C, mAP))
