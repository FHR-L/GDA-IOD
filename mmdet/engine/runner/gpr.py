import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import quadprog
import torch
from mmengine.model import detect_anomalous_params
from mmengine.runner import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals
from torch import Tensor
from torch.overrides import handle_torch_function
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS


@LOOPS.register_module()
class GPR(BaseLoop):
    """Loop for epoch-based training.
   #
    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            is_use: bool=False,
            lamda: int=1.0,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.lamda = lamda
        self.is_use = is_use
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

        # self.runner.model._set_static_graph()

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        # self.runner.val_loop.run()

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        # if self._epoch == 0:
        #     self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def project_grad(self, paras, grad_old_cls, grad_new_cls):
        for para, grad_old, grad_new in zip(paras, grad_old_cls, grad_new_cls):
            if grad_old is not None and grad_new is not None:
                grad_old_norm = grad_old / torch.linalg.norm(grad_old)
                grad_new_norm = grad_new / torch.linalg.norm(grad_new)

                if torch.dot(grad_old_norm.flatten(), grad_new_norm.flatten()) < 0:
                    grad_proj_ = (grad_new - torch.dot(
                        grad_new.flatten(), grad_old_norm.flatten()
                    ) * grad_old_norm * self.lamda)
                    # grad_proj_ = grad_proj_ * torch.linalg.norm(grad_new) / torch.linalg.norm(grad_proj_)
                    para.grad.data.copy_(grad_proj_ + para.grad)
                else:
                    para.grad.data.copy_(grad_new + para.grad)
            elif grad_new is not None:
                para.grad.data.copy_(grad_new + para.grad)

    def store_origin_grad(self, paras, grad_new_cls):
        for para, grad_new in zip(paras, grad_new_cls):
            if grad_new is not None:
                para.grad.data.copy_(grad_new)

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.

        # outputs = self.runner.model.train_step(
        #     data_batch, optim_wrapper=self.runner.optim_wrapper)

        with self.runner.optim_wrapper.optim_context(self):
            data = self.runner.model.module.data_preprocessor(data_batch, training=True)
            losses = self.runner.model._run_forward(data, mode='loss')

        """
        self.runner.model 是MMDistributedDataParallel封装了的，MMDistributedDataParallel重写了
        train_step()方法，因此base_model中的train_step()方法不会生效。

        """
        (loss_old_positive_cls, loss_old_positive_bbox,
         loss_new_positive_cls, loss_new_positive_bbox,
         loss_shared, log_vars) = self.runner.model.module.parse_losses_v3(losses)
        # -------------------------------------------------------
        # 替换：self.runner.optim_wrapper.update_params(parsed_loss)
        step_kwargs = {}
        zero_kwargs = {}
        loss_old_positive_cls = self.runner.optim_wrapper.scale_loss(loss_old_positive_cls)
        loss_old_positive_bbox = self.runner.optim_wrapper.scale_loss(loss_old_positive_bbox)
        loss_new_positive_cls = self.runner.optim_wrapper.scale_loss(loss_new_positive_cls)
        loss_new_positive_bbox = self.runner.optim_wrapper.scale_loss(loss_new_positive_bbox)
        loss_shared = self.runner.optim_wrapper.scale_loss(loss_shared)
        # print("loss old:{}, loss_new:{}, loss shared:{}".format(loss_old_positive, loss_new_positive, loss_shared))

        ori_model = self.runner.model.module

        self.runner.optim_wrapper.zero_grad(**zero_kwargs)
        if loss_old_positive_cls + loss_old_positive_bbox == 0:
            paras = []
            for para in ori_model.parameters():
                if para.requires_grad:
                    paras.append(para)
            self.runner.optim_wrapper.backward(loss_old_positive_cls + loss_old_positive_bbox + loss_new_positive_cls + loss_new_positive_bbox + loss_shared)
        else:
            paras = []
            for para in ori_model.parameters():
                if para.requires_grad:
                    paras.append(para)
            # cls
            grad_old_cls = torch.autograd.grad(loss_old_positive_cls + loss_old_positive_bbox, paras, retain_graph=True, allow_unused=True)
            grad_new_cls = torch.autograd.grad(loss_new_positive_cls + loss_old_positive_cls +
                                               loss_new_positive_bbox + loss_old_positive_bbox + loss_shared,
                                               paras,
                                               retain_graph=True,
                                               allow_unused=True)
            if self.is_use:
                try:
                    self.project_grad(paras, grad_old_cls, grad_new_cls)
                except:
                    print("none type, is use")
            else:
                try:
                    self.store_origin_grad(paras, grad_new_cls)
                except:
                    print("none type, is use")
            self.runner.optim_wrapper.backward(0.0 * (loss_new_positive_cls + loss_old_positive_cls +
                                                        loss_new_positive_bbox + loss_old_positive_bbox + loss_shared))


    # Update parameters only if `self._inner_count` is divisible by
    # `self._accumulative_counts` or `self._inner_count` equals to
    # `self._max_counts`
        if self.runner.optim_wrapper.should_update():
            # self.runner.optim_wrapper.step(**step_kwargs)
            if self.runner.optim_wrapper.clip_grad_kwargs:
                self.runner.optim_wrapper._clip_grad()
            self.runner.optim_wrapper.optimizer.step(**step_kwargs)
            self.runner.optim_wrapper.zero_grad(**zero_kwargs)

        # -------------------------------------------------------
        if self.runner.model.detect_anomalous_params:
            detect_anomalous_params(loss_old_positive_cls + loss_old_positive_bbox
                                    + loss_new_positive_cls + loss_new_positive_bbox
                                    + loss_shared, model=self)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=log_vars)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]

    def store_grad(self, pp, grads, grad_dims, tid):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
            tid: task id
        """
        # store the gradients
        grads[:, tid].fill_(0.0)
        cnt = 0
        for grad in pp:
            if grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en, tid].copy_(grad.data.view(-1))
            cnt += 1

    def overwrite_grad(self, pp, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp:
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1

    def project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))

    def project2o(self, gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().contiguous().view(-1).double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        dotp = np.sum(np.multiply(gradient_np, memories_np))
        ref_mag = np.sum(np.multiply(memories_np, memories_np))
        proj = gradient_np - ((dotp / ref_mag) * memories_np)

        gradient.copy_(torch.Tensor(proj).view(-1, 1))

    def overwrite_grad_add(self, pp, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp:
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad + param.grad)
            cnt += 1