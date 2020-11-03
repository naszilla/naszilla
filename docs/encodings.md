# A Study on Encodings for Neural Architecture Search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

[A Study on Encodings for Neural Architecture Search](https://arxiv.org/abs/2007.04965)\
Colin White, Willie Neiswanger, Sam Nolen, and Yash Savani.\
_arxiv:2007.04965_.

Many algorithms for neural architecture search (NAS) represent each neural architecture in the search space as a directed acyclic graph (DAG), and then search over all DAGs by encoding the adjacency matrix and list of operations as a set of hyperparameters. Recent work has demonstrated that even small changes to the way each architecture is encoded can have a significant effect on the performance of NAS algorithms. We present the first formal study on the effect of architecture encodings for NAS.
<p align="center">
  <img src="../img/encodings.png" alt="encodings" width="90%">
</p>

## Installation
See the [main readme file](https://github.com/naszilla/nas_encodings/blob/master/README.md) for installation instructions.

#### Download index-hash
Some of the path-based encoding methods require a hash map from path indices to cell architectures. We have created a pickle file which contains this hash map (size 57MB), located [here](https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing). Place it in the top level folder of this repo.

## Run encodings experiments
```bash
python run_experiments.py --algo_params evo_encodings --search_space nasbench_101
```
This command will run evolutionary search with six different encodings. To customize your experiments, open up `params.py`.

## Citation
Please cite [our paper](https://arxiv.org/abs/2007.04965) if you use code from this repo:

```bibtex
@inproceedings{white2020study,
  title={A Study on Encodings for Neural Architecture Search},
  author={White, Colin and Neiswanger, Willie and Nolen, Sam and Savani, Yash},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
