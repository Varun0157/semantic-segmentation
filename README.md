Semantic Segmentation using FCN and UNet models, done as a part of Assignment 4 of the Computer Vision course (IIIT-Hyderabad, Spring '25). The assignment details can be found in the [assignment document](./docs/CV_S25_A4.pdf).

## Results

### FCN

To train and test the model, run: 
```sh
python -m src.main <options>
```
The options include the data directory, batch size, model variant, frozen vs unfrozen backbone, the number of epochs, and the learning rate. 

To visualise some test results on a checkpoint, run: 
```sh
python -m src.visualise <options>
```
The options include the data directory, batch size (every item in a batch is visualised), and the model variant. 

### UNet

To train and test the model, run: 
```sh
python -m src.main <options>
```
The options include the data directory, batch size, model variant, frozen vs unfrozen backbone, the number of epochs, and the learning rate. 

To visualise some test results on a checkpoint, run: 
```sh
python -m src.visualise <options>
```
The options include the data directory, batch size (every item in a batch is visualised), and the model variant. 


## Setup

Download the data from [the following link](https://drive.google.com/drive/folders/1s2ZgwawnZyZXc5eei5cWmgV2A7UXOrMV?usp=sharing).

The environment can be set up with the conda env file:

```sh
cd docs
conda env create -f env.yml
```

or

```sh
pip install -r requirements.txt
```

Alternatively, install the dependencies as in [the conda history](./docs/env-hist.yml).

### Scripts
Helper utilities are available to split the train data [into train and validation](./q1/scripts/split.py) as well as to perform some [exploratory data analysis](./q1/scripts/eda.py) in both sub-parts. 
