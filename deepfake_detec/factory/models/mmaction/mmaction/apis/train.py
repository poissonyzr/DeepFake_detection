from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel
from mmcv.parallel import MMDistributedDataParallel

from mmaction.core import (DistOptimizerHook, DistEvalTopKAccuracyHook,
                           AVADistEvalmAPHook)
from mmaction.datasets import build_dataloader
from .env import get_root_logger


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars,
        num_samples=len(data['img_group_0'].data))

    return outputs


def train_network(model,
                  dataset,
                  cfg,
                  distributed=False,
                  validate=False,
                  logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if cfg.data.val.type in ['RawFramesDataset', 'VideoDataset']:
            runner.register_hook(
                DistEvalTopKAccuracyHook(cfg.data.val, k=(1, 5)))
        if cfg.data.val.type == 'AVADataset':
            runner.register_hook(AVADistEvalmAPHook(cfg.data.val))
    # if validate:
    #     if isinstance(model.module, RPN):
    #         # TODO: implement recall hooks for other datasets
    #         runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
    #     else:
    #         if cfg.data.val.type == 'CocoDataset':
    #             runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
    #         else:
    #             runner.register_hook(DistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)