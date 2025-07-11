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

You can run pascol_voc_split.py to split the Pascal VOC dataset as you want,

```python
# to generate instances_train2017_sel_last_40_cats.json
python pascol_voc_split.py

```


### Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in '/dataset/coco/'

# train first 40 cats
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_first_40_cats.py 2 --work-dir=../ERD_results/gfl_increment/gfl_r50_fpn_1x_coco_first_40_cats
#train last 40 cats incrementally
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py 2 --work-dir=../ERD_results/gfl_increment/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats
```
