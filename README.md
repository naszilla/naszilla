# naszilla
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

A repository to compare many popular NAS algorithms seamlessly across three popular benchmarks (NASBench 101, 201, and 301). You can implement your own NAS algorithm, and then easily compare it with ten algorithms across three benchmarks.

This repository contains the official code for the following three papers, including a [NeurIPS2020 spotlight](https://arxiv.org/abs/2007.04965) paper:

<table>
 <tbody>
    <tr align="center" valign="bottom">
      <th>Paper</th>
      <th>Readme</th>
      <th>Blog Post</th>
    </tr>
    <tr> <!-- (1st row) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> <a href="https://arxiv.org/abs/2007.04965" target="_blank">A Study on Encodings for Neural Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/naszilla/nas_encodings/blob/master/docs/encodings.md">encodings.md</a> </td>
    <td align="center" valign="middle"> <a href="https://abacus.ai/blog/2020/10/02/a-study-on-encodings-for-nas/" target="_blank">Blog Post</a> </td>
    </tr>
    <tr> <!-- (2nd row) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> <a href="https://arxiv.org/abs/1910.11858" target="_blank">BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/naszilla/nas_encodings/blob/master/docs/bananas.md">bananas.md</a> </td>
    <td align="center" valign="middle"> <a href="https://medium.com/reality-engines/bananas-a-new-method-for-neural-architecture-search-192d21959c0c" target="_blank">Blog Post</a> </td>
    </tr>
    <tr> <!-- (3rd row) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> <a href="https://arxiv.org/abs/2005.02960" target="_blank">Local Search is State of the Art for Neural Architecture Search Benchmarks</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/naszilla/nas_encodings/blob/master/docs/local_search.md">local_search.md</a> </td>
    <td align="center" valign="middle"> <a href="https://abacus.ai/blog/local-search-is-state-of-the-art-for-neural-architecture-search-benchmarks/" target="_blank">Blog Post</a> </td>
    </tr>
 </tbody>
</table>

## Requirements
- tensorflow == 1.14.0
- nasbench (follow the installation instructions [here](https://github.com/google-research/nasbench))
- nas-bench-201 (follow the installation instructions [here](https://github.com/D-X-Y/NAS-Bench-201))
- nasbench301 (follow the installation instructions [here](https://github.com/automl/nasbench301))
- pybnn (used only for the DNGO baseline algorithm. Installation instructions [here](https://github.com/automl/pybnn))

#### Download nasbench datasets
- Download `nasbench_only108.tfrecord` (size 499MB) [here](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord)
- Download `NAS-Bench-201-v1_0-e61699.pth` (size 2.1GB) from [here](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs)
- Download `nasbench301_models_v0.9.zip` (size 1.58GB) from [here](https://figshare.com/articles/software/nasbench301_models_v0_9_zip/12962432)
- Place these files in one folder, e.g., `nas_benchmark_datasets` in your top-level directory.

## Run NAS experiments on NASBench-101/201/301 search spaces

```bash
python run_experiments.py --search_space nasbench_101 --queries 150 --trials 10
```
This will test several NAS algorithms against each other on the NASBench-101 search space.
To customize your experiment, open `params.py`. Here, you can change the algorithms to run and their hyperparameters.

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
