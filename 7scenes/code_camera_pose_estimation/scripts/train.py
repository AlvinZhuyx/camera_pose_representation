import set_paths
from common.train import Trainer
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion,\
  MapNetOnlineCriterion
from models.posenet import PoseNet, MapNet
from dataset_loaders.composite import MF, MFOnline
import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models

parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes'),
                    help='Dataset')
parser.add_argument('--datapath', type=str, default='../../dataset/7Scenes')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet'),
  help='Model to train')
parser.add_argument('--train', type=bool, default=False, help='retrain the model or test the trained checkpoints')
parser.add_argument('--device', type=str, default='0',
  help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
  default=None)
parser.add_argument('--learn_beta', action='store_true',
  help='Learn the weight of translation loss')
parser.add_argument('--learn_gamma', action='store_true',
  help='Learn the weight of rotation loss')
parser.add_argument('--resume_optim', action='store_true',
  help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
args = parser.parse_args()

assert args.model == 'posenet'
assert args.dataset == '7Scenes'

if not args.train:
    assert args.checkpoint is not None

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')

section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
sax = section.getfloat('sax')
saq = section.getfloat('beta')
print('dropout', dropout, 'color_jitter', color_jitter, 'sax', sax, 'saq', saq)
if args.model.find('mapnet') >= 0:
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  srx = 0.0
  srq = section.getfloat('gamma')
  steps = section.getint('steps')
if args.model.find('++') >= 0:
  vo_lib = section.get('vo_lib', 'orbslam')
  print('Using {:s} VO'.format(vo_lib))

section = settings['training']
seed = section.getint('seed')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# model
feature_extractor = models.resnet34(pretrained=True)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=True,
                  filter_nans=(args.model=='mapnet++'))
model = posenet


# loss function
train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=args.learn_beta)
val_criterion = PoseNetCriterion()

# optimizer
param_list = [{'params': model.parameters()}]
if args.learn_beta and hasattr(train_criterion, 'sax') and \
    hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
    hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, **optim_config)

data_dir = osp.join('..', 'data', args.dataset)
stats_file = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
crop_size_file = osp.join(data_dir, 'crop_size.txt')
crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))
# transformers
train_tforms = [transforms.Resize(256)]
if color_jitter > 0:
  assert color_jitter <= 1.0
  if args.train:
    print('Using ColorJitter data augmentation')
  train_tforms.append(transforms.ColorJitter(brightness=color_jitter,
    contrast=color_jitter, saturation=color_jitter, hue=0.5))
train_tforms.append(transforms.ToTensor())
train_tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
train_data_transform = transforms.Compose(train_tforms)

test_tforms = [transforms.Resize(256)]
test_tforms.append(transforms.ToTensor())
test_tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
test_data_transform = transforms.Compose(test_tforms)


target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# datasets
kwargs = dict(scene=args.scene, data_path=args.datapath, target_transform=target_transform, seed=seed)
from dataset_loaders.seven_scenes import SevenScenes
train_set = SevenScenes(train=True, transform=train_data_transform, **kwargs)
val_set = SevenScenes(train=False, transform=test_data_transform, **kwargs)


# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene,
  args.model, config_name)
if args.learn_beta:
  experiment_name = '{:s}_learn_beta'.format(experiment_name)
if args.learn_gamma:
  experiment_name = '{:s}_learn_gamma'.format(experiment_name)
experiment_name += args.suffix
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, device=args.device,
                  checkpoint_file=args.checkpoint,
                  resume_optim=args.resume_optim, val_criterion=val_criterion, train=args.train)
trainer.train_val(lstm=False)
