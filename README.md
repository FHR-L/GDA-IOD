### Install
You can follow the steps to prepare the environment:
```python
conda create -n CLOD python=3.8 -y

source activate CLOD

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install tqdm

pip install -U openmim

mim install mmengine==0.7.3

mim install mmcv==2.0.0

# cd GDA-COLD
pip install -v -e .
```
### Dataset Prepare
You can run pascol_voc_split.py to split the Pascal VOC dataset as you want,
```python
# modify in pascol_voc_split.py
dataset_root = r'data root to VOC'
# run
python pascol_voc_split.py
```
### Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and add dataset root to 'configs/_base_/datasets/voc/voc10_10.py' and 'configs/_base_/datasets/voc/voc10.py'

# train first 10 cats
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/faster-rcnn/10.py 1 --work-dirs ./work_dir/base/voc/10

#train last 10 cats incrementally
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/_base_/datasets/voc/voc10_10.py 1
```
### Citation
```
@inproceedings{gda_iod,
 author = {W. Luo and S. Zhang and D. Cheng and Y. Xing and G. Liang and P. Wang and Y. Zhang},
 title  = {Gradient Decomposition and Alignment for Incremental Object Detection},
 year   = {2025},
 booktitle = {ICCV}
}
```
