#!/usr/bin/env octave-cli -qf

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

clear param
param.imageSize = [32 32]; % set a normalized image size
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

if nargin < 1
  error("Usage: ./cifar10_gist.m filename \n")
endif
arg_list = argv();
name = arg_list{1};
if ~exist(name, "file")
  error("Input argument is not a valid filename")
endif
printf("Extracting gists from %s\n", name)

sz = 10000;
d = load(name);
Img = d.data;
batchGists = zeros(10000, 512);

for idx=1:sz
    red = Img(idx, 1:1024);
    green = Img(idx, 1025:2048);
    blue = Img(idx, 2049:3072);

    Image = zeros(32, 32, 3);
    Image(:,:,1) = reshape(red, 32, 32);
    Image(:,:,2) = reshape(green, 32, 32);
    Image(:,:,3) = reshape(blue, 32, 32);

    [gist_cifar, param]  =  LMgist ( Image , '' , param  );
    batchGists(idx,:) = gist_cifar;
    if mod(idx, 100) == 0
      printf("%s: %d/10000\n", name, idx)
    endif
end
save("-ascii", strrep(name, ".mat", "_gist.mat"), "batchGists");
labels = d.labels;
save("-ascii", strrep(name, ".mat", "_label.mat"), "labels");
