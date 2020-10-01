# A GPU benchmark with fastai
> A set of notebooks/scripts to bench GPU using fastai and Pytorch


This file will become your README and also the index of your documentation.

## Install

You should only need the latest fastai. Please create a clean new environment to launch the benchmarks.

You can run this repo directly after cloning it.

## How to use

you should run this a script (it uses fascript under the hood)

```bash
python bench.py --gpu 0
```
will run the benchs on the gpu 0. (default is running using `dataparallel` on all available GPUs)
