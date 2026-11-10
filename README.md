# Introduction

Training and evaluation scripts for SASM-grasp on the GraspNet-1B dataset.

# Usage

## Prepare Conda Env

```bash
conda create -n finegrasp python=3.10.12
conda activate finegrasp

## Install requirements

## Install torch, spconv that matches your cuda version, for example:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install spconv-cu118==2.3.8

## Option 1: Install robo_orchard_lab and finegrasp from internet
pip3 install robo_orchard_lab[finegrasp]

## Option 2: Install robo_orchard_lab and finegrasp from local
git clone https://github.com/ZZU-EIDM-Lab/SASM-Grasp.git
cd robo_orchard_lab && make version
pip3 install .[finegrasp]



## Install MinkowskiEngine and graspNetAPI
```bash
git clone https://github.com/graspnet/graspnetAPI.git && cd graspnetAPI
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip3 install .
pip3 install transforms3d==0.4.2 numpy==1.26.4

git clone https://github.com/NVIDIA/MinkowskiEngine.git && cd MinkowskiEngine
conda install openblas-devel -c anaconda
python3 setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

## Install pointnet2 and knn op

Compile and install pointnet2 and knn operators from [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp/tree/main)

```
git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp

cd Scale-Balanced-Grasp/pointnet2 && python3 setup.py install && cd ../..

cd Scale-Balanced-Grasp/knn && python3 setup.py install

```
## Prepare Dataset

For dataset basic preparation, please follow the instructions provided in [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp).


## check Envrionment
```bash
cd projects/finegrasp_graspnet1b
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace \
    --data_root ./data \
    --kwargs '{"train_split":"mini", "test_split":"mini", "test_mode":["test_scale_mini"], "step_log_freq":10, "max_epoch":1, "lr":0.0}'

```
## Single GPU Training

```bash
cd projects/finegrasp_graspnet1b
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace \
    --data_root ./data
```

## Multi GPU Training

```bash
cd projects/finegrasp_graspnet1b
accelerate launch \
    --multi-gpu \
    --num-processes 4 \
    --gpu-ids 0,1,2,3 \
    train.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace \
    --data_root ./data
```


## Eval
```bash
cd projects/finegrasp_graspnet1b
python3 eval.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace/eval_results
    --data_root ./data
```

## Quick inference using pipeline and pretrain models

```bash
cd projects/finegrasp_graspnet1b
python3 infer.py
```


We thank the following repositories for their valuable contributions:

- [GraspNetAPI](https://github.com/graspnet/graspnetAPI)
- [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp)
- [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp)
- [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)


