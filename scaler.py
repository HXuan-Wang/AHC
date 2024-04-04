# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import timm
import torch
import torch.distributed as dist
from timm.utils.clip_grad import dispatch_clip_grad





class NativeScaler_part_update(timm.utils.NativeScaler):

    def __init__(self):
        super().__init__()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm',parameters=None, update=True,create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)

        if update:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

