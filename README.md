# supervised-hashing-baselines

This repository contains code to reproduce the baselines in

 **<a href="https://arxiv.org/abs/1609.06753">How should we evaluate supervised hashing?</a>**
 <br>
 Alexandre Sablayrolles,
 Matthijs Douze,
 Nicolas Usunier,
 Hervé Jégou
 <br>
ICASSP 2017


If you find this code useful in your research then please cite
```
@article{sablayrolles2017supervisedhashing,
  title={How should we evaluate supervised hashing?},
  author={Sablayrolles, Alexandre and Douze, Matthijs and Usunier, Nicolas and Jégou, Hervé},
  booktitle = {2017 {IEEE} International Conference on Acoustics, Speech and Signal
               Processing, {ICASSP} 2017, New Orleans, LA, USA, March 5-9, 2017},
  pages     = {1732--1736},
  year={2017}
}
```

# Pre-requisites

Install [faiss](https://github.com/facebookresearch/faiss) and the requirements:
```
pip install requirements.txt
```


# Setup

The following code downloads the GIST descriptors extracted from the CIFAR-10 images.

```
./setup.sh  
```
To see how the features were extracted, you can take a look at the features/ folder, (you also need the [GIST extractor](http://people.csail.mit.edu/torralba/code/spatialenvelope/) from MIT).

# Demo

This code was developed and tested for MacOS 10.12 with Python 2.7.
The following script launches all experiments of table 1 on GIST descriptors.

```
./run.sh
```

Typical output:
```
Experiments with 1000 labels and 300 anchors:
One hot encoding: 0.267 mAP (C=16.00) (0.26672553388)
LSH: 0.295 mAP (C=8.00) (0.29518721115)
Topline: 0.337 mAP (C=8.00) (0.33676588748)

Experiments with 5000 labels and 1000 anchors:
One hot encoding: 0.373 mAP (C=64.00) (0.37287632364)
LSH: 0.419 mAP (C=32.00) (0.41861792332)
Topline: 0.480 mAP (C=32.00) (0.47958284901)

Experiment with 59000 labels and 1000 anchors: 0.755 mAP (C=2048.00) (0.75499136667)
```
