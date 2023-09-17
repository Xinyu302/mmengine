# Copyright (c) OpenMMLab. All rights reserved.

import os
import os.path as osp
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from mmengine.runner import Runner
from mmengine.model import BaseModel
from mmengine.hooks import RecorderHook
from mmengine.testing import RunnerTestCase

class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)
        
        self.OUTPUT_0 = 'runner_model:forward:outputs@0'
        self.OUTPUT_1 = 'runner_model:forward:outputs@1'
        self.WEIGHT = 'runner_model:forward:self.linear1.weight'
        self.recorder = {self.OUTPUT_0: [],
                         self.OUTPUT_1: [],
                         self.WEIGHT: []}

    def forward(self, inputs, data_samples, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_samples, list):
            data_sample = torch.stack(data_samples)
        
        outputs = self.linear1(inputs)
        self.recorder[self.OUTPUT_0].append(outputs)
        outputs = self.linear2(outputs)
        self.recorder[self.OUTPUT_1].append(outputs)
        self.recorder[self.WEIGHT] \
            .append(self.linear1.weight.detach().clone())

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_sample - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs

SAVE_DIR = '../tmp_dir/work_dir'
os.makedirs(SAVE_DIR, exist_ok=True)

class TestRecorderHook(RunnerTestCase):
    def test_init(self):
        # Test recorders
        with self.assertRaises(ValueError):
            RecorderHook(None, SAVE_DIR)
        with self.assertRaises(ValueError):
            RecorderHook([], SAVE_DIR)
        
        # Test target
        with self.assertRaises(ValueError):
            RecorderHook([dict(type='AttributeRecorder')], SAVE_DIR)
    
    def test_recorder(self):
        x = [(torch.ones(2, 2), [torch.ones(2, 1)])]
        # train_dataset = [x, x, x]
        train_dataset = x * 50
        train_dataloader = DataLoader(train_dataset, batch_size=2)

        model = ToyModel()
        runner = Runner(
            model=model,
            custom_hooks=[
                dict(
                    type='RecorderHook',
                    recorders=[
                        dict(type='FunctionRecorder', target='outputs', index=[0, 1]),
                        dict(type='AttributeRecorder', target='self.linear1.weight')
                    ],
                    save_dir=SAVE_DIR,
                    print_modification=True)
            ],
            work_dir='../tmp_dir',
            train_dataloader=train_dataloader,
            train_cfg=dict(by_epoch=True, max_epochs=10),
            optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
        runner.train()

        for i in range(1, 11):
            self.assertTrue(osp.isfile(osp.join(SAVE_DIR, f'record_epoch_{i}.pth')))

        for i in range(1, 11):        
            tensor = torch.load(osp.join(SAVE_DIR, f'record_epoch_{i}.pth'))

            for j in range(0, 25):
                self.assertTrue(model.recorder[model.OUTPUT_0][(i - 1) * 25 + j].equal(tensor[model.OUTPUT_0][j]))
                self.assertTrue(model.recorder[model.OUTPUT_1][(i - 1) * 25 + j].equal(tensor[model.OUTPUT_1][j]))
                self.assertTrue(model.recorder[model.WEIGHT][(i - 1) * 25 + j].equal(tensor[model.WEIGHT][j]))