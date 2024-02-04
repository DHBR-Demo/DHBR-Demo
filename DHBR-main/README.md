# README

Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with Intel(R) Xeon(R) Gold 6248R CPU @3.00GHz and single GPU NVIDIA GeForce RTX 3090.

## Environment

```
conda env create -f environment.yaml
```

## Running

Train DHBR on the dataset Youshu with GPU 0:
```
python train.py -g 0 -m DHBR -d Youshu
```

## Files Definition

- ```datasets``` : contains three public datasets: Youshu, NetEase, iFashion
- ```models``` : contains python files of our framework
    - ```DHBR.py``` : model file
- ```config.yaml``` : config file
- ```train.py``` : train the model
- ```utility.py``` : data preprocess
- ```checkpoints``` : saves the best model and associate config file for each experiment setting
- ```log``` : saves the experimental logs in txt-format
- ```runs``` : saves the experimental logs in tensorboard-format
