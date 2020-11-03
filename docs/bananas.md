# BANANAS
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

[BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search](https://arxiv.org/abs/1910.11858)\
Colin White, Willie Neiswanger, and Yash Savani.\
_arXiv:1910.11858_.

## A new method for neural architecture search
BANANAS is a neural architecture search (NAS) algorithm which uses Bayesian optimization with a meta neural network to predict the validation accuracy of neural architectures. We use a path-based encoding scheme to featurize the neural architectures that are used to train the neural network model. After training on just 200 architectures, we are able to predict the validation accuracy of new architectures to within one percent on average. The full NAS algorithm beats the state of the art on the NASBench and the DARTS search spaces. On the NASBench search space, BANANAS is over 100x more efficient than random search, and 3.8x more efficent than the next-best algorithm we tried. On the DARTS search space, BANANAS finds an architecture with a test error of 2.57%.

<p align="center">
<img src="../img/bananas.png" alt="bananas" width="70%">
</p>

## Installation
See the [main readme file](https://github.com/naszilla/nas_encodings/blob/master/README.md) for installation instructions.

## Run BANANAS experiments

```bash
python run_experiments.py --search_space nasbench_101 --queries 150 --trials 10
```
This will compare BANANAS to other NAS algorithms on the NASBench-101 search space.
To customize your experiment, open `params.py`. Here, you can change the algorithms to run and their hyperparameters.

Note: to run experiments on the DARTS search space directly (without using nasbench-301), see the [original repository](https://github.com/naszilla/bananas). Warning: that experiment is compute-intensive.

<p align="center">
  <img src="../img/metann_adj_train.png" alt="adj_train" width="24%">
  <img src="../img/metann_adj_test.png" alt="adj_test" width="24%">
  <img src="../img/metann_path_train.png" alt="path_train" width="24%">
  <img src="../img/metann_path_test.png" alt="path_test" width="24%">
</p>

## Citation
Please cite [our paper](https://arxiv.org/abs/1910.11858) if you use code from this repo:

```bibtex
@article{white2019bananas,
  title={BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search},
  author={White, Colin and Neiswanger, Willie and Savani, Yash},
  journal={arXiv preprint arXiv:1910.11858},
  year={2019}
}
```

