# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import math
import os
from collections import OrderedDict
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from mmengine import Config, is_list_of
from mmengine.runner import load_checkpoint, load_state_dict
from mmengine.structures import InstanceData
from scipy.stats import norm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from torch import Tensor, nn
from tqdm import tqdm

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import ForwardResults
from .two_stage import TwoStageDetector
from ...apis import inference_detector, init_detector
from ...structures import SampleList, OptSampleList
import einops
import time
import xml.etree.ElementTree as ET
from scipy.stats import beta
from scipy.optimize import minimize

@MODELS.register_module()
class FasterRCNNIncrease(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 pseudo_label_setting: ConfigType,
                 ori_setting: ConfigType,
                 current_dataset_setting: ConfigType,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.num_classes = ori_setting['num_classes']
        self.ori_num_classes = None
        self.load_base_detector(ori_setting)
        self.use_weight_pseudo_label = pseudo_label_setting['is_use']

        # print('self.')

        if self.use_weight_pseudo_label:
            self.pseudo_label_alpha = pseudo_label_setting['alpha']
            if current_dataset_setting['dataset'] == 'coco':
                self.initial_gauss_mixture_model_coco(current_dataset_setting, ori_setting)
            else:
                self.initial_gauss_mixture_model_voc(current_dataset_setting, ori_setting)
        else:
            self.pseudo_label_threshold = pseudo_label_setting['threshold']
        torch.save(self.state_dict(), ori_setting['load_from_weight'])

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1.cpu().tolist()
        x1_2, y1_2, x2_2, y2_2 = box2.cpu().tolist()
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area
        return iou

    def calculate_iou_cpu(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1.cpu().tolist()
        x1_2, y1_2, x2_2, y2_2 = box2
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area
        return iou

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        gt_pseudo_num = []
        gt_new_num = []
        result_ori = self.ori_model.predict(batch_inputs, copy.deepcopy(batch_data_samples), rescale=False)
        pseudo_label_cls_weights_list = []
        pseudo_label_reg_weights_list = []
        rpn_data_samples_only_new_exclude_old = copy.deepcopy(batch_data_samples)
        rpn_data_samples_only_old_exclude_new = copy.deepcopy(batch_data_samples)

        if self.use_weight_pseudo_label:
            for result, batch_data_sample, batch_data_sample_only_new in zip(result_ori, batch_data_samples, rpn_data_samples_only_new_exclude_old):
                pseudo_label_cls_weights = []
                pseudo_label_reg_weights = []
                indices = cv2.dnn.NMSBoxes(
                    result.pred_instances['bboxes'].cpu().numpy(),
                    result.pred_instances['scores'].cpu().numpy(),
                    0.,
                    0.5)
                for bbox_index, bbox in enumerate(result.pred_instances):
                    if bbox_index not in indices:
                        continue
                    filter_tag = 0
                    confidence = bbox['scores'].item()

                    if confidence < self.negative_threshold:
                        continue
                    elif confidence > self.positive_threshold:
                        for ground_true_bbox in batch_data_sample.gt_instances['bboxes']:
                            iou = self.calculate_iou(ground_true_bbox, bbox['bboxes'][0])
                            if iou > 0.7:
                                filter_tag = 1
                        if filter_tag == 1:
                            continue
                        cls_weight = 1.
                        reg_weight = 1.
                        pseudo_label_cls_weights.append(cls_weight)
                        pseudo_label_reg_weights.append(reg_weight)
                        bbox.__delattr__('scores')
                        batch_data_sample.gt_instances = batch_data_sample.gt_instances.cat(
                            [batch_data_sample.gt_instances, bbox])
                        batch_data_sample_only_new.ignored_instances = batch_data_sample_only_new.ignored_instances.cat(
                            [batch_data_sample_only_new.ignored_instances, bbox]
                        )
                    else:
                        cls_weight = 0.
                        reg_weight = 0.
                        pseudo_label_cls_weights.append(cls_weight)
                        pseudo_label_reg_weights.append(reg_weight)
                        bbox.__delattr__('scores')
                        batch_data_sample.gt_instances = batch_data_sample.gt_instances.cat(
                            [batch_data_sample.gt_instances, bbox])
                        batch_data_sample_only_new.ignored_instances = batch_data_sample_only_new.ignored_instances.cat(
                            [batch_data_sample_only_new.ignored_instances, bbox]
                        )

                pseudo_label_cls_weights_list.append(pseudo_label_cls_weights)
                pseudo_label_reg_weights_list.append(pseudo_label_reg_weights)
                gt_pseudo_num.append(len(pseudo_label_reg_weights))
                gt_new_num.append(len(batch_data_sample.gt_instances) - len(pseudo_label_reg_weights))
        else:
            for result, batch_data_sample in zip(result_ori, batch_data_samples):
                indices = cv2.dnn.NMSBoxes(
                    result.pred_instances['bboxes'].cpu().numpy(),
                    result.pred_instances['scores'].cpu().numpy(),
                    0.,
                    0.7)
                pseudo_label_cls_weights = []
                pseudo_label_reg_weights = []
                for bbox_index, bbox in enumerate(result.pred_instances):
                    if bbox['scores'] < 0.5:
                        continue
                    if bbox_index not in indices:
                        continue
                    max_iou =0
                    for ground_true_bbox in batch_data_sample.gt_instances['bboxes']:
                        iou = self.calculate_iou(ground_true_bbox, bbox['bboxes'][0])
                        if iou > max_iou:
                            max_iou = iou
                    if max_iou > 0.7:
                        cls_weight = 0.3
                        reg_weight = 0.3
                        pseudo_label_cls_weights.append(cls_weight)
                        pseudo_label_reg_weights.append(reg_weight)
                        bbox.__delattr__('scores')
                        batch_data_sample.gt_instances = batch_data_sample.gt_instances.cat(
                            [batch_data_sample.gt_instances, bbox])
                        continue
                    else:
                        bbox.__delattr__('scores')
                        pseudo_label_cls_weights.append(1.0)
                        pseudo_label_reg_weights.append(1.0)
                        batch_data_sample.gt_instances = batch_data_sample.gt_instances.cat(
                            [batch_data_sample.gt_instances, bbox])
                pseudo_label_reg_weights_list.append(pseudo_label_reg_weights)
                pseudo_label_cls_weights_list.append(pseudo_label_cls_weights)
                gt_pseudo_num.append(len(pseudo_label_reg_weights))
                gt_new_num.append(len(batch_data_sample.gt_instances) - len(pseudo_label_reg_weights))
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)
            # add
            for rpn_data_sample_only_new_exclude_old, rpn_data_sample_only_old_exclude_new in \
                zip(rpn_data_samples_only_new_exclude_old, rpn_data_samples_only_old_exclude_new):
                rpn_data_sample_only_old_exclude_new.ignored_instances = rpn_data_sample_only_new_exclude_old.gt_instances
                rpn_data_sample_only_old_exclude_new.gt_instances = rpn_data_sample_only_new_exclude_old.ignored_instances
                rpn_data_sample_only_old_exclude_new.gt_instances.labels = \
                    torch.zeros_like(rpn_data_sample_only_old_exclude_new.gt_instances.labels)
            rpn_losses_old, _ = self.rpn_head.loss_and_predict(
                x, rpn_data_samples_only_new_exclude_old, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses_old.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses_old[f'rpn_{key}_old'] = rpn_losses_old.pop(key)
            # add end

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss_weight(x, rpn_results_list,
                                               batch_data_samples,
                                               pseudo_label_cls_weights_list,
                                               pseudo_label_reg_weights_list,
                                               gt_pseudo_num,
                                               gt_new_num,
                                               self.ori_num_classes,
                                               self.num_classes)

        losses.update(roi_losses)

        return losses

    def parse_losses_v3(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss_old_positive_cls = sum(value for key, value in log_vars if ('loss' in key and 'old_positive' in key and 'cls' in key))
        loss_old_positive_bbox = sum(value for key, value in log_vars if ('loss' in key and 'old_positive' in key and 'bbox' in key))
        loss_new_positive_cls = sum(value for key, value in log_vars if ('loss' in key and 'new_positive' in key and 'cls' in key))
        loss_new_positive_bbox = sum(value for key, value in log_vars if ('loss' in key and 'new_positive' in key and 'bbox' in key))
        loss_shared = sum(value for key, value in log_vars if (('loss' in key)
                                                               and ('old_positive' not in key)
                                                               and ('new_positive' not in key)))
        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return (loss_old_positive_cls, loss_old_positive_bbox,
                loss_new_positive_cls, loss_new_positive_bbox,
                loss_shared, log_vars)  # type: ignore


    def parse_voc_xml(self, xml_file_path):
        """
        解析PASCAL VOC格式的XML文件，返回包含对象类别的列表以及边界框的列表。

        参数:
        xml_file_path -- XML文件的路径

        返回:
        labels -- 包含对象类别的列表
        bboxes -- 包含边界框位置的列表，格式为[xmin, ymin, xmax, ymax]
        """
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        labels = []
        bboxes = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            labels.append(label)
            bboxes.append([xmin, ymin, xmax, ymax])

        return labels, bboxes

    def initial_gauss_mixture_model_voc(self, current_dataset_setting, ori_setting, model_type='GMM'):
        class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                       'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
        print("==================Initial Gauss Mixture Models ing====================")
        see_class = class_names[self.ori_num_classes: self.num_classes]
        if current_dataset_setting['cache_path'] is not None and os.path.exists(current_dataset_setting['cache_path']):
            print("use confidence cache from: {}".format(current_dataset_setting['cache_path']))
            labels_confidences = torch.load(current_dataset_setting['cache_path'])
        else:
            cfg_path = ori_setting['ori_config_file']
            pt_path = ori_setting['ori_checkpoint_file']
            model = init_detector(cfg_path, pt_path)
            train_val = current_dataset_setting['train_val']
            img_dir = current_dataset_setting['img_dir']
            xml_dir = '/home/Newdisk/luowenlong/Datasets/VOC2007/Annotations'
            labels_confidences = []

            for i in range(ori_setting['ori_num_classes']):
                labels_confidences.append([])
            with open(train_val, 'r', encoding='utf-8') as file:
                files = []
                for line in file:
                    files.append(line.strip())
                for line in tqdm(files):
                    filename = line.strip()
                    img_file = os.path.join(img_dir, filename + '.jpg')
                    xml_file = os.path.join(xml_dir, filename + '.xml')
                    g_labels, g_bboxes = self.parse_voc_xml(xml_file)

                    result = inference_detector(model, img_file)
                    result = result.pred_instances
                    for label, score, bbox in zip(result.labels, result.scores, result.bboxes):
                        skip_tag = False
                        for g_bbox, g_label in zip(g_bboxes, g_labels):
                            if g_label in see_class:
                                iou = self.calculate_iou_cpu(bbox, g_bbox)
                                if iou > 0.8:
                                    skip_tag = True
                        if skip_tag:
                            continue
                        label = int(label)
                        score = float(score)
                        labels_confidences[label].append(score)
            if current_dataset_setting['cache_path'] is not None:
                torch.save(labels_confidences, current_dataset_setting['cache_path'])

        self.negative_threshold = 0.
        self.positive_threshold = 0.
        self.gauss_model = GaussianMixture(n_components=2, max_iter=5000, random_state=1234, covariance_type='tied')
        confidences_concate = []
        for confidence in labels_confidences:
            confidences_concate.extend(confidence)
        train_data = torch.Tensor(confidences_concate).reshape(-1, 1)
        self.gauss_model.fit(train_data)
        means = self.gauss_model.means_
        covs = self.gauss_model.covariances_
        self.negative_threshold = (means[1] + means[0]) / 2.
        if means[0] > means[1]:
            self.positive_threshold = means[0] - self.pseudo_label_alpha * (math.sqrt(covs[0]))
        else:
            self.positive_threshold = means[1] - self.pseudo_label_alpha * (math.sqrt(covs[0]))
        print("GMM negative_threshold: {}, positive_threshold: {}".format(self.negative_threshold, self.positive_threshold))

    def initial_gauss_mixture_model_coco(self, current_dataset_setting, ori_setting):
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                       'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                       'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                       'toothbrush']
        print("==================Initial Gauss Mixture Models ing====================")
        if os.path.exists(current_dataset_setting['cache_path']):
            print("use confidence cache from: {}".format(current_dataset_setting['cache_path']))
            labels_confidences = torch.load(current_dataset_setting['cache_path'])
        else:
            cfg_path = ori_setting['ori_config_file']
            pt_path = ori_setting['ori_checkpoint_file']
            model = init_detector(cfg_path, pt_path)
            train_val = current_dataset_setting['train_val']
            img_dir = current_dataset_setting['img_dir']
            self.use_class_special = current_dataset_setting['use_class_special']

            labels_confidences = []

            for i in range(ori_setting['ori_num_classes']):
                labels_confidences.append([])
            with open(train_val, 'r') as f:
                coco_data = json.load(f)

            images = coco_data['images']
            for image in tqdm(images):

                filename = image['file_name']
                img_file = os.path.join(img_dir, filename)
                result = inference_detector(model, img_file)
                result = result.pred_instances
                for label, score, bbox in zip(result.labels, result.scores, result.bboxes):
                    label = int(label)
                    score = float(score)
                    labels_confidences[label].append(score)
            torch.save(labels_confidences, current_dataset_setting['cache_path'])

        self.gauss_model = GaussianMixture(n_components=2, max_iter=5000, random_state=1234, covariance_type='tied')
        # print(labels_confidences[0])
        confidence_concate = []
        for data in zip(labels_confidences):
            confidence_concate.extend(data)
        train_data = torch.Tensor(data).reshape(-1, 1)
            # print(train_data.size())
        self.gauss_model.fit(train_data)
        print("==================Initial Gauss Mixture Models over====================")
        means = self.gauss_model.means_
        covs = self.gauss_model.covariances_
        negative_threshold = (min(means) + self.pseudo_label_alpha * math.sqrt(covs[0]))[0]
        positive_threshold = (max(means) - self.pseudo_label_alpha * math.sqrt(covs[0]))[0]
        print(means)
        print(covs)
        print("negative_threshold: {}, positive_threshold: {}".format(negative_threshold, positive_threshold))

        # self.positive_index = []
        # # positive index
        # for i in range(ori_setting['ori_num_classes']):
        #     output = self.gauss_model_list[i].predict_proba(torch.tensor([[1.0]]))
        #     index0 = output[0, 0]
        #     index1 = output[0, 1]
        #     if index0 > index1:
        #         self.positive_index.append([0, 1])
        #     else:
        #         self.positive_index.append([1, 0])
        # # examination


    def get_positive_and_negative_score(self, output, class_index):
        positive_index, negative_index = self.positive_index[class_index]
        return output[0, positive_index], output[0, negative_index]


    # def parse_losses(
    #     self, losses: Dict[str, torch.Tensor]
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    #     """Parses the raw outputs (losses) of the network.
    #
    #     Args:
    #         losses (dict): Raw output of the network, which usually contain
    #             losses and other necessary information.
    #
    #     Returns:
    #         tuple[Tensor, dict]: There are two elements. The first is the
    #         loss tensor passed to optim_wrapper which may be a weighted sum
    #         of all losses, and the second is log_vars which will be sent to
    #         the logger.
    #     """
    #     log_vars = []
    #     for loss_name, loss_value in losses.items():
    #         if isinstance(loss_value, torch.Tensor):
    #             log_vars.append([loss_name, loss_value.mean()])
    #         elif is_list_of(loss_value, torch.Tensor):
    #             log_vars.append(
    #                 [loss_name,
    #                  sum(_loss.mean() for _loss in loss_value)])
    #         else:
    #             raise TypeError(
    #                 f'{loss_name} is not a tensor or list of tensors')
    #
    #     loss_old_positive = sum(value for key, value in log_vars if ('loss' in key and 'old_positive' in key))
    #     loss_new_positive = sum(value for key, value in log_vars if ('loss' in key and 'new_positive' in key))
    #     loss_shared = sum(value for key, value in log_vars if (('loss' in key)
    #                                                            and ('old_positive' not in key)
    #                                                            and ('new_positive' not in key)))
    #     loss = sum(value for key, value in log_vars if 'loss' in key)
    #     log_vars.insert(0, ['loss', loss])
    #     log_vars = OrderedDict(log_vars)  # type: ignore
    #
    #     return loss_old_positive, loss_new_positive, loss_shared, log_vars  # type: ignore

    def parse_losses_v2(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss_old_positive = sum(value for key, value in log_vars if ('loss' in key and 'old_positive' in key))
        loss_new_positive = sum(value for key, value in log_vars if ('loss' in key and 'new_positive' in key))
        loss_shared = sum(value for key, value in log_vars if (('loss' in key)
                                                               and ('old_positive' not in key)
                                                               and ('new_positive' not in key)))
        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss_old_positive, loss_new_positive, loss_shared, log_vars  # type: ignore

    def soft_weight_v1(self, mean_var, score):
        mean_p = mean_var[0]
        var_p = mean_var[1] * self.pseudo_label_alpha
        mean_n = mean_var[2]
        var_n = mean_var[3] * self.pseudo_label_alpha
        prob_negative = 1 - norm.cdf(score, mean_n, var_n)
        prob_positive = norm.cdf(score, mean_p, var_p)
        if prob_negative > prob_positive:
            weight = 0
        else:
            weight = prob_positive * self.pseudo_label_alpha
        return float(weight)

    def soft_weight_v2(self, mean_var, score):
        mean_p = mean_var[0]
        var_p = mean_var[1] * self.pseudo_label_alpha

        weight = norm.cdf(score, mean_p, var_p)
        return float(weight * self.pseudo_label_alpha)

    def load_base_detector(self, ori_setting):
        """
                Initialize detector from config file.
        :param ori_setting:
        :return:
        """
        assert os.path.isfile(ori_setting['ori_checkpoint_file']), '{} is not a valid file'.format(
            ori_setting['ori_checkpoint_file'])
        ##### init original model & frozen it #####
        # build model
        ori_cfg = Config.fromfile(ori_setting['ori_config_file'])
        if hasattr(ori_cfg.model, 'latest_model_flag'):
            ori_cfg.model.latest_model_flag = False
        ori_model = MODELS.build(ori_cfg.model)
        # load checkpoint
        load_checkpoint(ori_model, ori_setting.ori_checkpoint_file, strict=False)
        # # set to eval mode
        # ori_model = init_detector(ori_setting['ori_config_file'], ori_setting.ori_checkpoint_file)

        ori_model.eval()
        ori_model.ori_model = None
        # ori_model.forward = ori_model.forward_dummy
        # # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        # ##### init original branchs of new model #####
        self.ori_num_classes = ori_setting.ori_num_classes
        self._load_checkpoint_for_new_model_v3(ori_setting.ori_checkpoint_file, strict=False)
        print('======> load base checkpoint for new model from {}'.format(ori_setting.ori_checkpoint_file))
        self.ori_model = ori_model

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=True, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                                       v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = self.roi_head.bbox_head.fc_cls.weight[self.ori_num_classes:, ...]
        added_branch_bias = self.roi_head.bbox_head.fc_cls.bias[self.ori_num_classes:]
        state_dict['roi_head.bbox_head.fc_cls.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_cls.weight'][:self.ori_num_classes], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_cls.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_cls.bias'][:self.ori_num_classes], added_branch_bias), dim=0)

        # modify reg head size of state_dict
        added_branch_weight = self.roi_head.bbox_head.fc_reg.weight[self.ori_num_classes * 4:, ...]
        added_branch_bias = self.roi_head.bbox_head.fc_reg.bias[self.ori_num_classes * 4:]
        state_dict['roi_head.bbox_head.fc_reg.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.weight'], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_reg.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.bias'], added_branch_bias), dim=0)

        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)

    def _load_checkpoint_for_new_model_v2(self, checkpoint_file, map_location=None, strict=True, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                                       v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = self.roi_head.bbox_head.fc_cls.weight[self.ori_num_classes:-1, ...]
        added_branch_bias = self.roi_head.bbox_head.fc_cls.bias[self.ori_num_classes:-1]
        state_dict['roi_head.bbox_head.fc_cls.weight'] = torch.cat(
            (
                state_dict['roi_head.bbox_head.fc_cls.weight'][:self.ori_num_classes],
                added_branch_weight,
                state_dict['roi_head.bbox_head.fc_cls.weight'][-1:]
            ), dim=0)
        state_dict['roi_head.bbox_head.fc_cls.bias'] = torch.cat(
            (
                state_dict['roi_head.bbox_head.fc_cls.bias'][:self.ori_num_classes],
                added_branch_bias,
                state_dict['roi_head.bbox_head.fc_cls.bias'][-1:]
            ), dim=0)

        # modify reg head size of state_dict
        added_branch_weight = self.roi_head.bbox_head.fc_reg.weight[self.ori_num_classes * 4:, ...]
        added_branch_bias = self.roi_head.bbox_head.fc_reg.bias[self.ori_num_classes * 4:]
        state_dict['roi_head.bbox_head.fc_reg.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.weight'], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_reg.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.bias'], added_branch_bias), dim=0)

        state_dict_new = {}
        for k, v in state_dict.items():
            if k.startswith('ori_model'):
                continue
            state_dict_new[k] = v

        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict_new, strict, logger)
        else:
            load_state_dict(self, state_dict_new, strict, logger)

    def _load_checkpoint_for_new_model_v3(self, checkpoint_file, map_location=None, strict=True, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                                       v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = (torch.mean(state_dict['roi_head.bbox_head.fc_cls.weight'][:self.ori_num_classes], dim=0, keepdim=True)).expand(self.num_classes - self.ori_num_classes, -1)

        added_branch_bias = (torch.mean(state_dict['roi_head.bbox_head.fc_cls.bias'][:self.ori_num_classes], dim=0, keepdim=True)).expand(self.num_classes - self.ori_num_classes)

        added_branch_weight = added_branch_weight.expand(self.num_classes - self.ori_num_classes, -1)
        added_branch_bias.expand(self.num_classes - self.ori_num_classes)
        state_dict['roi_head.bbox_head.fc_cls.weight'] = torch.cat(
            (
                state_dict['roi_head.bbox_head.fc_cls.weight'][:self.ori_num_classes],
                added_branch_weight,
                state_dict['roi_head.bbox_head.fc_cls.weight'][-1:]
            ), dim=0)
        state_dict['roi_head.bbox_head.fc_cls.bias'] = torch.cat(
            (
                state_dict['roi_head.bbox_head.fc_cls.bias'][:self.ori_num_classes],
                added_branch_bias,
                state_dict['roi_head.bbox_head.fc_cls.bias'][-1:]
            ), dim=0)

        # modify reg head size of state_dict
        added_branch_weight = state_dict['roi_head.bbox_head.fc_reg.weight']
        added_branch_bias = state_dict['roi_head.bbox_head.fc_reg.bias']
        added_branch_weight = torch.mean(einops.rearrange(added_branch_weight, '(n m) d-> n m d', m=4), dim=0, keepdim=True)
        added_branch_bias = torch.mean(einops.rearrange(added_branch_bias, '(n m)-> n m', m=4),dim=0, keepdim=True)

        added_branch_weight = added_branch_weight.expand(self.num_classes - self.ori_num_classes, -1, -1)
        added_branch_bias = added_branch_bias.expand(self.num_classes - self.ori_num_classes, -1)

        added_branch_weight = einops.rearrange(added_branch_weight, 'n m d -> (n m) d')
        added_branch_bias = einops.rearrange(added_branch_bias, 'n m-> (n m)')

        # print(self.roi_head.bbox_head.fc_reg.weight.shape)
        # print(self.roi_head.bbox_head.fc_reg.bias.shape)
        # assert 1 < 0
        state_dict['roi_head.bbox_head.fc_reg.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.weight'], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_reg.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.bias'], added_branch_bias), dim=0)

        state_dict_new = {}
        for k, v in state_dict.items():
            if k.startswith('ori_model'):
                continue
            state_dict_new[k] = v

        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict_new, strict, logger)
        else:
            load_state_dict(self, state_dict_new, strict, logger)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        # assert 1 < 0
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        # assert 1 < 0
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

