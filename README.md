# Cortex

Cortex is a minimal deep training/evaluation engine built upon PyTorch. It follows a very simple deep learning coding paradigm, but is flexible enough to reproduce algorithms from a variety of areas.

*(Currently we have released a [metric learning benchmark](cortex/apps/metric_learning/README.md) written using `cortex`. We're going to release more later after thorough tests)*


## Features of `cortex`

1. __It supports distributed and mixed-precision GPU training.__ Simply set `distributed=True` and/or `mixed_precision=True` to enable them.
2. __It simplifies the development of deep learning models.__ `cortex` does most trivial things such as `logging, checkpointing, training/evaluation looping, device allocating, optimizing,` and `scheduling`, while you'll only need to __implement 5 (sometimes 6) functions__ to define a model:
    - `build_dataloader`
    - `build_training`
    - `train_step`
    - `val_step`
    - `test_step`
    - `optimize_step` _# optional for, e.g., GANs and meta learners_
3. __It reproduces a large number of deep learning algorithms using the the unified interfaces.__ Currently we have released the [metric learning benchmark](cortex/apps/metric_learning/README.md). We'll released more benchmarks on detection, tracking, GANs, etc. later after thorough tests.
4. __It is light-weighted.__ The (training/evaluation) engine contains only two major functions: one calls a single or multiple processes defined in the other.

Check `cortex/apps` for a set of examples showing how to use `cortex` in your research.


# Install

First install PyTorch, faiss, scipy, and apex (optional, if you want to use mixed-precision training), then run:

```bash
git clone https://github.com/huanglianghua/cortex.git
cd cortex
pip install -e .
```
