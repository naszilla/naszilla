# Local Search for NAS
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

[Local Search is State of the Art for Neural Architecture Search Benchmarks](https://arxiv.org/abs/2005.02960)\
Colin White, Sam Nolen, and Yash Savani.\
_arXiv:2005.02960_.

We study the simplest versions of local search, showing that local search achieves strong results on NASBench-101 (size 10^6) and NASBench-201 (size 10^4). See our paper for a theoretical study which characterizes the performance of local search on graph optimization problems, backed by simulation results.

<p align="center">
  <img src="../img/local_search.png" alt="local_search" width="70%">
</p>
In the left figure, each point is an architecture from NAS-Bench-201 trained on CIFAR10, and each edge denotes the LS function. We plotted the trees of the nine architectures with the lowest test losses. The right figure is similar, but the architectures are assigned validation losses at random. We see that we are much more likely to converge to an architecture with low loss on structured data (CIFAR10) rather than unstructured (random) data.

## Installation
See the [main readme file](https://github.com/naszilla/nas_encodings/blob/master/README.md) for installation instructions.

## Run local search experiments on NASBench-101/201/301 search spaces
```bash
python run_experiments.py --algo_params local_search_variants --search_space nasbench_101 --queries 150 --trials 10
```
This will test a few simple variants of local search against a few other NAS algorithms. To customize your experiment, open `params.py`. Here, you can change the algorithms to run and their hyperparameters. To run with nas-bench-201, add the flag `--search_space nasbench_201 --dataset cifar10`  to the above command, where the dataset can be set to `cifar10`, `cifar100`, or `imagenet`.

## Citation
Please cite [our paper](https://arxiv.org/abs/2005.02960) if you use code from this repo:
```bibtex
@article{white2020local,
  title={Local Search is State of the Art for Neural Architecture Search Benchmarks},
  author={White, Colin and Nolen, Sam and Savani, Yash},
  journal={arXiv preprint arXiv:2005.02960},
  year={2020}
}
```
