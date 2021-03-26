"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
pytorch data loader for the 7-scenes dataset
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data
from .utils import load_image
import sys
import pickle
import transforms3d.euler as txe
import transforms3d.quaternions as txq

sys.path.insert(0, '../')
from common.pose_utils import process_poses

class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, real=False,
                 skip_images=False, seed=7):
      """
      :param scene: scene name ['chess', 'pumpkin', ...]
      :param data_path: root 7scenes data directory.
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the images
      :param target_transform: transform to apply to the poses
      :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
      :param real: If True, load poses from SLAM/integration of VO
      :param skip_images: If True, skip loading images and return None instead
      """
      self.mode = mode
      self.transform = transform
      self.target_transform = target_transform
      self.skip_images = skip_images

      base_dir = data_dir = osp.join(data_path, scene)
      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'TrainSplit.txt')
      else:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}
      vo_stats = {}
      gt_offset = int(0)
      for seq in seqs:
        seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
        seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
        if real:
          raise NotImplementedError
        else:
          frame_idx = np.array(np.arange(len(p_filenames)), dtype=np.int)

          pss = []
          for i in frame_idx:
            tmp_p = np.zeros(96 * 2 + 7)
            tmp_embed = np.load(osp.join(seq_dir, 'frame_{:06d}_posvec.npy'.format(i)))
            assert tmp_embed.shape == (96 * 2,)
            tmp_p[:192] = tmp_embed
            tmp_pose = np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:12]
            cur_min = np.squeeze(np.load(os.path.join(base_dir, 'min_loc_all.npy')))
            tmp_loc = tmp_pose[[3, 7, 11]] - cur_min
            tmp_pose = np.reshape(tmp_pose, (3, 4))[:3, :3]
            tmp_p[-7:-4] = tmp_loc
            tmp_p[-4:] = txq.mat2quat(tmp_pose)
            pss.append(tmp_p)
          ps[seq] = np.asarray(pss)

          vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
        gt_offset += len(p_filenames)
        c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                  for i in frame_idx]
        d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                  for i in frame_idx]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)

      pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
      if train and not real:
        mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
        std_t = np.ones(3)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
      else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

      self.poses = np.empty((0, 96 * 2 + 7))
      for seq in seqs:
        self.poses = np.vstack((self.poses, ps[seq]))

    def __getitem__(self, index):
      assert self.mode == 0
      if self.skip_images:
        img = None
        pose = self.poses[index]
      else:
        if self.mode == 0:
          img = None
          while img is None:
            img = load_image(self.c_imgs[index])
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 1:
          img = None
          while img is None:
            img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 2:
          c_img = None
          d_img = None
          while (c_img is None) or (d_img is None):
            c_img = load_image(self.c_imgs[index])
            d_img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
          img = [c_img, d_img]
          index -= 1
        else:
          raise Exception('Wrong mode {:d}'.format(self.mode))

      if self.target_transform is not None:
        pose = self.target_transform(pose)

      if self.skip_images:
        return img, pose

      if self.transform is not None:
        if self.mode == 2:
          img = [self.transform(i) for i in img]
        else:
          img = self.transform(img)

      return img, pose

    def __len__(self):
      return self.poses.shape[0]

