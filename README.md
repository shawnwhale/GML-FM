# Enhancing Factorization Machines with Generalized Metric Learning
This is the official implementation for our TKDE paper titled with `Enhancing Factorization Machines with Generalized 
Metric Learning`. Please follow the flags and directory settings in `config.py`.

## Repo-download
```
git clone https://github.com/guoyang9/GML-FM.git
```
## Prerequisites
    * python==3.7.7
    * tqdm==4.31.1
    * numpy==1.18.4
    * pytorch==1.4.0
    * tensorboardX==2.1
    * torchvision==0.6.0
    
## Dataset
We use 6 datasets in this paper including three Amazon dataset and one MovieLens dataset
and four of them are publicly available.

## Pre-processing

1. process the Amazon dataset:
    ```
    python pre-process/process_amazon.py
    ```
1. process the MovieLens dataset:
    ```
    python pre-process/process_ml-1m.py
    ```
    
## Model Training and Auto Evaluation
```
python main.py --lr 0.001 --batch-size 256 --gpu 0
```

## Citation
If you want to use this code, please cite our paper as follows:
```
@article{gml-fm,
  title={Enhancing Factorization Machines with Generalized Metric Learning},
  author={Yangyang Guo, Zhiyong Cheng, Jiazheng Jing, Yanpeng Lin, Liqiang Nie, Meng
Wang},
  journal={TKDE}
  year={2020},
  organization={IEEE}
}

``` 
