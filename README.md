# SwitchOut for Neural Machine Translation (InsightAI Project)

This project was created as part of the Insight AI Fellowship. I would like to thank my technical advisors Emmanuel Ameisen, Matt Rubashkin and Ming Zhao for their help in scoping the project and getting over the hurdles.

This repository contains an implementation of [Switchout](https://arxiv.org/abs/1808.07512) applied to train the Transfomer model described in [Attention is All You Need](https://arxiv.org/abs/1706.03762).

The transformer model used here is adapted from [ The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). We follow the set up described here to run the code on a [p3.8xlarge AWS instance](https://aws.amazon.com/ec2/instance-types/p3/)


## Motivation:
The authors demonstrate performance gains on three machine translation datasets using SwitchOut.
SwitchOut can also be adapted to other tasks, such as sentiment analysis, by modifying the existing implementation to only augment text and leave the labels unchanged.


## Requisites and Instructions to get started
Clone the repository
```
https://github.com/nsapru/SwitchOut.git
```

Create virutal environment
```
conda create -n switchout_venv python=3.6 anaconda
source activate switchout_venv
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
pip install numpy matplotlib spacy torchtext
```
It's recommended that you use the version of pytorch listed above.
The code will run significantly slower on AWS p3.8xlarge instances if you use the same version of pytorch (0.3.0.post4) compiled with earlier version of CUDA. Newer version of pytorch might require some code modifications in the training (train.py) as well as the model (transformer.py) file.


## Data
The dataset used to train and test the model is the German(de) to English(en) [IWSLT dataset] (https://torchtext.readthedocs.io/en/latest/datasets.html#iwslt) available with torchtext. This is also one of the datasets used by the authors to test SwitchOut. When you start training from the first time, the dataset will be downnloaded and pre-processed in the .data folder and will be available for training in subsequent sessions.

## Training and Inference
To train and test, simply run:
```
python train.py
```

## Run Inference
At this time, training in inference occurs in the same (train.py) file.
I am planning create a separate file, translate.py, to run inference from the command line using pre-trained copies of my model.
