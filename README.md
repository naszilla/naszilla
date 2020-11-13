<p align="center"><img src="img/naszilla_banner.png" width=700 /></p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

A repository to compare many popular NAS algorithms seamlessly across three popular benchmarks (NASBench 101, 201, and 301). You can implement your own NAS algorithm, and then easily compare it with ten algorithms across three benchmarks.

This repository contains the official code for the following three papers, including a [NeurIPS2020 spotlight](https://arxiv.org/abs/2007.04965) paper:

<table>
 <tbody>
    <tr align="center" valign="bottom">
      <th>Paper</th>
      <th>README</th>
      <th>Blog Post</th>
    </tr>
    <tr> <!-- (1st row) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> <a href="https://arxiv.org/abs/2007.04965" target="_blank">A Study on Encodings for Neural Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="docs/encodings.md">encodings.md</a> </td>
    <td align="center" valign="middle"> <a href="https://abacus.ai/blog/2020/10/02/a-study-on-encodings-for-nas/" target="_blank">Blog Post</a> </td>
    </tr>
    <tr> <!-- (2nd row) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> <a href="https://arxiv.org/abs/1910.11858" target="_blank">BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="docs/bananas.md">bananas.md</a> </td>
    <td align="center" valign="middle"> <a href="https://medium.com/reality-engines/bananas-a-new-method-for-neural-architecture-search-192d21959c0c" target="_blank">Blog Post</a> </td>
    </tr>
    <tr> <!-- (3rd row) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> <a href="https://arxiv.org/abs/2005.02960" target="_blank">Local Search is State of the Art for Neural Architecture Search Benchmarks</a> </td>
    <td align="center" valign="middle"> <a href="docs/local_search.md">local_search.md</a> </td>
    <td align="center" valign="middle"> <a href="https://abacus.ai/blog/local-search-is-state-of-the-art-for-neural-architecture-search-benchmarks/" target="_blank">Blog Post</a> </td>
    </tr>
 </tbody>
</table>

## Installation
First clone this repository and install its requirements
```
git clone https://github.com/naszilla/naszilla
cd naszilla
pip install -r requirements.txt
cd ..
```
Next, install nasbench
```
git clone https://github.com/google-research/nasbench
cd nasbench
pip install -e .
cd ..
```
Next, install nasbench301 (currently the pip version has an error)
```
git clone https://github.com/automl/nasbench301
cd nasbench301
cat requirements.txt | xargs -n 1 -L 1 pip install
export PYTHONPATH="${PYTHONPATH}:$PWD"
cd ..
```
Finally, download the nas benchmark datasets (either with the terminal commands below, or from their respective websites ([nasbench101](https://github.com/google-research/nasbench), [nasbench201](https://github.com/D-X-Y/NAS-Bench-201), and [nasbench301](https://github.com/automl/nasbench301)).
```
# these files are 0.5GB, 2.1GB, and 1.6GB, respectively
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
wget https://ndownloader.figshare.com/files/24693026 -O nasbench301_models_v0.9.zip
unzip nasbench301_models_v0.9.zip
gdrive download 16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_
# place all of them in one folder, e.g., ~/nas_benchmark_datasets
```
Now you have successfully installed all of the requirements to run eleven NAS algorithms on three benchmark datasets!

## Run NAS experiments on NASBench-101/201/301 search spaces

```bash
cd naszilla
python naszilla/run_experiments.py --search_space nasbench_101 --queries 100 --trials 10
```
This will test several NAS algorithms against each other on the NASBench-101 search
space.  To customize your experiment, open `params.py`. Here, you can change the
algorithms and their hyperparameters. For details on running specific methods,
see [these docs](docs/naszilla.md).

## Contributions
Contributions are welcome!

## Citation
Please cite our papers if you use code from this repo:
```bibtex
@inproceedings{white2020study,
  title={A Study on Encodings for Neural Architecture Search},
  author={White, Colin and Neiswanger, Willie and Nolen, Sam and Savani, Yash},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

@article{white2019bananas,
  title={BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search},
  author={White, Colin and Neiswanger, Willie and Savani, Yash},
  journal={arXiv preprint arXiv:1910.11858},
  year={2019}
}

@article{white2020local,
  title={Local Search is State of the Art for Neural Architecture Search Benchmarks},
  author={White, Colin and Nolen, Sam and Savani, Yash},
  journal={arXiv preprint arXiv:2005.02960},
  year={2020}
}
```

## Contents

This repo contains [encodings](docs/encodings.md) for neural architecture search, a
variety of NAS methods (including [BANANAS](docs/bananas.md), a neural predictor
Bayesian optimization method, and [local search](docs/local_search.md) for NAS), and an
easy interface for using multiple NAS benchmarks.

Encodings:
<p align="center">
  <img src="img/encodings.png" alt="encodings" width="90%">
</p>

BANANAS:
<p align="center">
  <img src="img/metann_adj_train.png" alt="adj_train" width="24%">
  <img src="img/metann_adj_test.png" alt="adj_test" width="24%">
  <img src="img/metann_path_train.png" alt="path_train" width="24%">
  <img src="img/metann_path_test.png" alt="path_test" width="24%">
</p>

Local search:
<p align="center">
  <img src="img/local_search.png" alt="local_search" width="65%">
</p>
