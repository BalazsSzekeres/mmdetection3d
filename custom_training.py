from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

# set config file
config_file = osp.join('.', 'configs', 'pointpillars', 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py')
cfg = Config.fromfile(config_file)

# set work_dir
cfg.work_dir = './work_dirs\point_pillar_test'

# set gpu ids
cfg.gpu_ids = [0]
# cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
distributed = False

# create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

# # dump config
# cfg.dump('.\work_dirs\point_pillar_test')

# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

# TODO: ugly workaround to judge whether we are training det or seg model
if cfg.model.type in ['EncoderDecoder3D']:
    logger_name = 'mmseg'
else:
    logger_name = 'mmdet'
logger = get_root_logger(
    log_file=log_file, log_level=cfg.log_level, name=logger_name)

# init the meta dict to record some important information such as
# environment info and seed, which will be logged
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
meta['env_info'] = env_info
meta['config'] = cfg.pretty_text

# log some basic info
logger.info(f'Distributed training: {distributed}')
logger.info(f'Config:\n{cfg.pretty_text}')

# set random seeds
seed = init_random_seed(1)
logger.info(f'Set random seed to {seed}')
cfg.seed = seed

# building model
model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
model.init_weights()

logger.info(f'Model:\n{model}')

datasets = [build_dataset(cfg.data.train)]

if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    # in case we use a dataset wrapper
    if 'dataset' in cfg.data.train:
        val_dataset.pipeline = cfg.data.train.dataset.pipeline
    else:
        val_dataset.pipeline = cfg.data.train.pipeline
    # set test_mode=False here in deep copied config
    # which do not affect AP/AR calculation later
    # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
    val_dataset.test_mode = False
    datasets.append(build_dataset(val_dataset))
if cfg.checkpoint_config is not None:
    # save mmdet version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(
        mmdet_version=mmdet_version,
        mmseg_version=mmseg_version,
        mmdet3d_version=mmdet3d_version,
        config=cfg.pretty_text,
        CLASSES=datasets[0].CLASSES,
        PALETTE=datasets[0].PALETTE  # for segmentors
        if hasattr(datasets[0], 'PALETTE') else None)
# add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

train_model(
    model,
    datasets,
    cfg,
    distributed=distributed,
    validate=True,
    timestamp=timestamp,
    meta=meta)
