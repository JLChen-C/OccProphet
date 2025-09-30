<div align="center">

<img src="assets/occprophet_logo.png" alt="OccProphet Logo" class="occprophet-logo" style="width: 75%;">

## OccProphet: Pushing Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with Observer-Forecaster-Refiner Framework

<p align="center">
  <a href="https://jlchen-c.github.io/OccProphet/">
    <img src="https://img.shields.io/badge/OccProphet-Project_Page-_?labelColor=F9F2FE&color=yellow"></a>&nbsp;
  <a href="https://arxiv.org/abs/2502.15180">
    <img src="https://img.shields.io/badge/Arxiv-_?label=OccProphet&labelColor=F9F2FE&color=red"></a>&nbsp;
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/JLChen-C/OccProphet?labelColor=F9F2FE"></a>
</p>

</div>

## 🔍 Overview

OccProphet is a camera-only 4D occupancy forecasting framework, offering high efficiency in both training and inference, with excellent forecasting performance.

OccProphet has the following features:
- **Flexibility**: OccProphet only relies on camera inputs, making it flexible and can be easily adapted to different traffic scenarios.
- **High Efficiency**: OccProphet is both training- and inference-friendly, with a lightweight Observer-Forecaster-Refiner pipeline.
- **High Performance**: OccProphet achieves state-of-the-art performance on three real-world 4D occupancy forecasting datasets: nuScenes, Lyft-Level5 and nuScenes-Occupancy.



## 🔥 Latest News

- [2025/10/01] Code and checkpoints of OccProphet are released.

## 🔧 Installation

We follow the installation instructions in [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc).


* Create and activate a conda environment:
```bash
conda create -n occprophet python=3.7 -y
conda activate occprophet
```

* Install PyTorch
```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

* Install GCC-6

```bash
conda install -c omgarcia gcc-6
```

* Install MMCV, MMDetection, and MMSegmentation
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install yapf==0.40.1
```

* Install MMDetection3D
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
```

* Install other dependecies
```bash
pip install timm==0.9.12 huggingface-hub==0.16.4 safetensors==0.4.2
pip install open3d-python==0.7.0.0
pip install PyMCubes==0.1.4
pip install spconv-cu113
pip install fvcore
pip install setuptools==59.5.0
```

* Install Lyft-Level5 dataset SDK
```bash
pip install lyft_dataset_sdk
```

* Install OccProphet
```bash
cd ..
git clone https://github.com/JLChen-C/OccProphet.git
cd OccProphet
export PYTHONPATH="."
python setup.py develop
export OCCPROPHET_DIR="$(pwd)"
```

Optional: If you encounter issues for training or inference on GMO + GSO tasks, follow the instructions below to fix the issues

* Install Numba and LLVM-Lite
```bash
pip install numba==0.55.0
pip install llvmlite==0.38.0

# Reinstall setuptools if you encounter this issue: AttributeError: module 'distutils' has no attribute 'version'
# pip install setuptools==59.5.0
```

* Modify the files:
```
In Line 5, file $PATH_TO_ANACONDA/envs/occprophet/lib/python3.7/site-packages/mmdet3d-0.17.1-py3.7-linux-x86_64.egg/mmdet3d/datasets/pipelines/data_augment_utils.py:

Replace
"from numba.errors import NumbaPerformanceWarning"
with
"from numba.core.errors import NumbaPerformanceWarning"

In Line 30, $PATH_TO_ANACONDA/envs/occprophet/lib/python3.7/site-packages/nuscenes/eval/detection/data_classes.py

Replace
"self.class_names = self.class_range.keys()"
with
"self.class_names = list(self.class_range.keys())"
```

* Install dependecies for visualization
```bash
sudo apt-get install Xvfb
pip install xvfbwrapper
pip install mayavi
```

## 📚 Dataset Preparation

* Create your data folder ```$DATA``` and download the datasets below to ```$DATA```
  - [nuScenes V1.0 full](https://www.nuscenes.org/nuscenes#download) dataset
  - [nuScenes-Occupancy](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing) dataset, and pickle files [nuscenes_occ_infos_train.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl) and [nuscenes_occ_infos_val.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl)
  - [Lyft-Level5](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data) dataset

* Link the datasets to the OccProphet folder
```bash
mkdir $OCCPROPHET_DIR/data
ln -s $DATA/nuscenes $OCCPROPHET_DIR/data/nuscenes
ln -s $DATA/nuscenes-occupancy $OCCPROPHET_DIR/data/nuscenes-occupancy
ln -s $DATA/lyft $OCCPROPHET_DIR/data/lyft
```

* Move the pickle files **nuscenes_occ_infos_train.pkl** and **nuscenes_occ_infos_val.pkl** to nuscenes dataset root:
```bash
mv $DATA/nuscenes_occ_infos_train.pkl $DATA/nuscenes/nuscenes_occ_infos_train.pkl
mv $DATA/nuscenes_occ_infos_val.pkl $DATA/nuscenes/nuscenes_occ_infos_val.pkl
```

* The dataset structure should be organized as the file tree below:
```bash
OccProphet
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   ├── nuscenes_occ_infos_val.pkl
│   ├── nuScenes-Occupancy/
│   ├── lyft/
│   │   ├── maps/
│   │   ├── train_data/
│   │   ├── images/   # from train images, containing xxx.jpeg
│   ├── cam4docc
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── GMO_lyft/
│   │   │   ├── ...
│   │   ├── MMO_lyft/
│   │   │   ├── ...
```

* The datas generation pipeline of GMO, GSO, and other tasks is integrated in the dataloader. You can directly run the training and evaluation scripts. It may take several hours for generation of each task during the first epoch, the following epochs will be much faster.
  - You can also generate the dataset without any training or inference by setting `only_generate_dataset = True` in the config file, or just adding `--cfg-options model.only_generate_dataset=True` after your command.

## 🧪 Training

* To launch the training, change your working directory to ```$OCCPROPHET_DIR``` and run the following command:
```bash
CUDA_VISIBLE_DEVICES=$YOUR_GPU_IDS PORT=$PORT bash run.sh $CONFIG $NUM_GPUS
```
  - Argument explanation:
&nbsp;&nbsp;&nbsp;&nbsp;`$YOUR_GPU_IDS`: The GPU ids you want to use
&nbsp;&nbsp;&nbsp;&nbsp;`$PORT`: The connection port of distributed training
&nbsp;&nbsp;&nbsp;&nbsp;`$CONFIG`: The config path
&nbsp;&nbsp;&nbsp;&nbsp;`$NUM_GPUS`: The number of available GPUs

For example, you can launch the training on GPUs 0, 1, 2, and 3 with the config file `./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py` as follows:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26000 bash run.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py 4
```

* Optional: The default is set to $2\times$ `data.samples_per_gpu` for faster data loading. If the training is stopped due to out of cpu memory, you can try to set the `data.workers_per_gpu=1` in the config file, or just adding `--cfg-options data.workers_per_gpu=1` after your command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26000 bash run.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py 4 --cfg-options data.workers_per_gpu=1
```


## 🔬 Evaluation

To launch the evaluation, change your working directory to `$OCCPROPHET_DIR` and run the following command:
```bash
CUDA_VISIBLE_DEVICES=$YOUR_GPU_IDS PORT=$PORT bash run_eval.sh $CONFIG $CHECKPOINT $NUM_GPUS --evaluate
```
* Argument explanation:
&nbsp;&nbsp;&nbsp;&nbsp;`$YOUR_GPU_IDS`: The GPU ids you want to use
&nbsp;&nbsp;&nbsp;&nbsp;`$PORT`: The connection port of distributed evaluation
&nbsp;&nbsp;&nbsp;&nbsp;`$CONFIG`: The config path
&nbsp;&nbsp;&nbsp;&nbsp;`$CHECKPOINT`: The checkpoint path
&nbsp;&nbsp;&nbsp;&nbsp;`$NUM_GPUS`: The number of available GPUs

* For example, you can launch the evaluation on GPUs 0, 1, 2, and 3 with the config file `./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py` as follows:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26006 bash run_eval.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py ./work_dirs/occprophet/OccProphet_4x1_inf-GMO_nuscenes/OccProphet_4x1_inf-GMO_nuscenes.pth 4
```

* The default evaluation measure the IoU of all future frames, you can change the evaluated time horizon by modifying the following settings in the config file or just adding them after your command.

  - For example, if you want to evaluate the IoU of the present frame, you can set `model.test_present=True` in the config file, or just adding `--cfg-options model.test_present=True` after your command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26006 bash run_eval.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py ./work_dirs/occprophet/OccProphet_4x1_inf-GMO_nuscenes/OccProphet_4x1_inf-GMO_nuscenes.pth 4 --cfg-options model.test_present=True
```

* Fine-grained Evaluation: you can evaluate the IoU of the X-th frame by setting `model.test_time_indices=X` in the config file, or just adding `--cfg-options model.test_time_indices=X` after your command.<br>
For example, if you want to evaluate the IoU of the 5-th frame from the last, you can set `model.test_time_indices=-5` in the config file, or just adding `--cfg-options model.test_time_indices=-5` after your command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26006 bash run_eval.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py ./work_dirs/occprophet/OccProphet_4x1_inf-GMO_nuscenes/OccProphet_4x1_inf-GMO_nuscenes.pth 4 --cfg-options model.test_time_indices=-5
```

* Additional: If you want to save the prediction results to `YOUR_RESULT_DIR`, you can set `model.save_pred=True model.save_path=YOUR_RESULT_DIR` in the config file, or just adding `--cfg-options model.save_pred=True model.save_path=YOUR_RESULT_DIR` after your command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26006 bash run_eval.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py ./work_dirs/occprophet/OccProphet_4x1_inf-GMO_nuscenes/OccProphet_4x1_inf-GMO_nuscenes.pth 4 --cfg-options model.save_pred=True model.save_path=YOUR_RESULT_DIR
```

* Optional: The default `data.workers_per_gpu` is set to $2\times$ `data.samples_per_gpu` for faster data loading. If the training is stopped due to out of cpu memory, you can try to set the `data.workers_per_gpu=1` in the config file, or just adding `--cfg-options data.workers_per_gpu=1` after your command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=26006 bash run_eval.sh ./projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py ./work_dirs/occprophet/OccProphet_4x1_inf-GMO_nuscenes/OccProphet_4x1_inf-GMO_nuscenes.pth 4 --cfg-options data.workers_per_gpu=1
```

Please note the the `VPQ` metric of 3D instance prediction are the raw model outputs **without refinement** proposed in [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc).


## 🖥️ Visualization

**Ground-Truth Occupancy Labels:** To visualize the ground-truth occupancy labels, you can run the following command to save the visualization results to `$SAVE_DIR` (default is `$OCCPROPHET_DIR/viz`), where `$PRED_DIR` is the directory of the prediction results:

```bash
cd $OCCPROPHET_DIR/viz
python viz_pred.py --pred_dir $PRED_DIR --save_dir $SAVE_DIR
```

* If you want to visualize the changes across different frames, you can add `--show_by_frame` after your command:
```bash
cd $OCCPROPHET_DIR/viz
python viz_pred.py --pred_dir $PRED_DIR --save_dir $SAVE_DIR --show_by_frame
```

**Occupancy Forecasting Results:** To visualize the occupancy forecasting results in `$PRED_DIR`, you can run the following command to save the visualization results to `$SAVE_DIR`:

```bash
cd $OCCPROPHET_DIR/viz
python viz_pred.py --pred_dir $PRED_DIR --save_dir $SAVE_DIR
```

* If you want to visualize the changes across different frames, you can add `--show_by_frame` after your command:
```bash
cd $OCCPROPHET_DIR/viz
python viz_pred.py --pred_dir $PRED_DIR --save_dir $SAVE_DIR --show_by_frame
```

## 📦 Checkpoint Release

OccProphet supports all 5 tasks:
- Inflated GMO on nuScenes dataset
- Inflated GMO on Lyft Level 5 dataset
- Fine-grained GMO on nuScenes-Occupancy dataset
- Inflated GMO and Fine-grained GSO on nuScenes and nuScenes-Occupancy datasets
- Fine-grained GMO and Fine-grained GSO on nuScenes-Occupancy dataset

The configs and checkpoints of all 5 tasks are released and can be accessed through the links in the table below:

| Task                               | Dataset                      | Config                                                                                                                                                       | Model                                                                                                                                                                                                |
|------------------------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Inflated GMO                       | nuScenes                     | [OccProphet_4x1_inf-GMO_nuscenes.py](projects/configs/occprophet/OccProphet_4x1_inf-GMO_nuscenes.py)                                                         | [OccProphet_4x1_inf-GMO_nuscenes.pth](https://github.com/JLChen-C/OccProphet/releases/download/Checkpoint/OccProphet_4x1_inf-GMO_nuscenes.pth)                                                       |
| Inflated GMO                       | Lyft-Level5                  | [OccProphet_4x1_inf-GMO_lyft-level5.py](projects/configs/occprophet/OccProphet_4x1_inf-GMO_lyft-level5.py)                                                   | [OccProphet_4x1_inf-GMO_lyft-level5.pth](https://github.com/JLChen-C/OccProphet/releases/download/Checkpoint/OccProphet_4x1_inf-GMO_lyft-level5.pth)                                                 |
| Fine-grained GMO                   | nuScenes-Occupancy           | [OccProphet_4x1_fine-GMO_nuscenes-occ.py](projects/configs/occprophet/OccProphet_4x1_fine-GMO_nuscenes-occ.py)                                               | [OccProphet_4x1_fine-GMO_nuscenes-occ.pth](https://github.com/JLChen-C/OccProphet/releases/download/Checkpoint/OccProphet_4x1_fine-GMO_nuscenes-occ.pth)                                             |
| Inflated GMO, Fine-grained GSO     | nuScenes, nuScenes-Occupancy | [OccProphet_fp16_4x1_inf-GMO+fine-GSO.py](projects/configs/occprophet/OccProphet_fp16_4x1_inf-GMO+fine-GSO.py)                                               | [OccProphet_fp16_4x1_inf-GMO+fine-GSO_nuscenes_nuscenes-occ.pth](https://github.com/JLChen-C/OccProphet/releases/download/Checkpoint/OccProphet_fp16_4x1_inf-GMO+fine-GSO_nuscenes_nuscenes-occ.pth) |
| Fine-grained GMO, Fine-grained GSO | nuScenes-Occupancy           | [OccProphet_fp16_4x1_fine-GMO+fine-GSO_nuscenes_nuscenes-occ.py](projects/configs/occprophet/OccProphet_fp16_4x1_fine-GMO+fine-GSO_nuscenes_nuscenes-occ.py) | [OccProphet_fp16_4x1_fine-GMO+fine-GSO_nuscenes-occ.pth](https://github.com/JLChen-C/OccProphet/releases/download/Checkpoint/OccProphet_fp16_4x1_fine-GMO+fine-GSO_nuscenes-occ.pth)                 |

## 📦 Citation

If you are interested in OccProphet, or find it useful to to your work, please feel free to give us a star ⭐ or cite our paper 😊:

```bibtex
@article{chen2025occprophet,
  title={Occprophet: Pushing efficiency frontier of camera-only 4d occupancy forecasting with observer-forecaster-refiner framework},
  author={Chen, Junliang and Xu, Huaiyuan and Wang, Yi and Chau, Lap-Pui},
  journal={arXiv preprint arXiv:2502.15180},
  year={2025}
}
```

## 🤝 Acknowledgement

We thank [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc) for their significant contribution to end-to-end 4D occupancy forecasting community. We develop our codebase upon their excellent work.

## 🌱 Related Works

- [3D-Occupancy-Perception](https://github.com/HuaiyuanXu/3D-Occupancy-Perception): A Survey on Occupancy Perception for Autonomous Driving: The Information Fusion Perspective