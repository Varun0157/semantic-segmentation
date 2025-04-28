Semantic Segmentation using FCN and UNet models, done as a part of Assignment 4 of the Computer Vision course (IIIT-Hyderabad, Spring '25). The assignment details can be found in the assignment document.

python -m src.main (options) in both

python -m src.visualise (options) in both

python scripts/split.py or eda.py if required

# Setup

Download the data from [the following link](https://drive.google.com/drive/folders/1s2ZgwawnZyZXc5eei5cWmgV2A7UXOrMV?usp=sharing).

The environment can be set up with the conda env file:

```sh
cd docs
conda env create -f env.yml
```

Alternatively, install the dependencies as in [the history of installs](./docs/env-hist.yml).

# TODO

- [x] prune data and checkpoints
- [ ] make nicer readme
