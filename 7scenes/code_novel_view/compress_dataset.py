from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import time
import os
from utils import *
import transforms3d.euler as txe


## This code does data preparation for training our novel view synthesis model
## 1. It translates the location of each scene into the [0,4] x [0,1.5] x [0,3] m^3 cubic (by substract the minimum value)
##    It calculates the minimum location value of each scene (individually) and save them to the "min_loc_all.npy" file in each scene directory
## 2. It compress all the data of one scene into a pickle file (e.g. 'chess/train.pickle' for training sequences of scene chess,
##    'chess/test.pickle' for testing sequences of chess). Our dataloader provided in the code will load from these pickle files to get
##    the data during training. Using this pickle file enable the novel view synthesis model to load in all the images of one scene at one time,
##    which reduce the overhead of loading the data during training.


im_sz = 128

path = '../dataset/7Scenes'
scenes = os.listdir(path)
scenes.sort()

start_time = time.time()
for s in scenes:
    if '.pickle' in s:
        continue
    print(s)
    scene_path = os.path.join(path, s)
    files = [('train.pickle', 'TrainSplit.txt'), ('test.pickle', 'TestSplit.txt')]
    all_locs = []
    tmp_dict = {}
    for tmp, file_name in files:
        print(file_name)
        index = None
        locs = None
        angles = None
        images = None
        total_data = 0.0
        tmp_path = os.path.join(scene_path, tmp)
        seqs = []
        with open(os.path.join(scene_path, file_name)) as f:
            line = f.readline()
            while line:
                seqs.append('seq-{:02d}'.format(int(line[8 : -1])))
                line = f.readline()
            for seq in seqs:
                print(seq)
                seq_path = os.path.join(scene_path, seq)
                pss = [n for n in os.listdir(seq_path) if 'pose.txt' in n]
                cur_locs = []
                cur_angles = []
                cur_images = []
                for i in range(len(pss)):
                    cur_img_file = 'frame-{:06d}.color.png'.format(i)
                    cur_pose_file = 'frame-{:06d}.pose.txt'.format(i)
                    img = load_rgb(os.path.join(seq_path, cur_img_file), im_sz)
                    ps = np.loadtxt(os.path.join(seq_path, cur_pose_file)).flatten()[:12]
                    loc = ps[[3, 7, 11]]
                    m = ps.reshape((3, 4))[:3, :3]
                    theta, phi, gamma = txe.mat2euler(m, 'sxyz')
                    theta += np.pi
                    phi += np.pi / 2
                    gamma += np.pi
                    cur_locs.append(loc)
                    cur_angles.append((theta, phi, gamma))
                    cur_images.append(img)
                cur_locs = np.array(cur_locs)
                cur_angles = np.array(cur_angles)
                cur_images = np.array(cur_images)
                if locs is None:
                    ## locs are the location information of each image;
                    ## angels are the campera orientation of each image;
                    ## images are resized to 128 x 128 for training novel view synthesis model
                    ## index is originally designed for our model to pick pairs of images that are close to each other;
                    ## At first, we only pick successive images and leave out the last image to prevent picking the images from two different seqs
                    ## Then we also tries to pick image pairs that has one or two images between them and we find that occasionally picking images
                    ## from two sequences does not influence the overall model training (since this happens with only a chance of 1/1000, and we see the rotation loss decrease well) so we just keep using this file.
                    locs = np.copy(cur_locs)
                    angles = np.copy(cur_angles)
                    images = np.copy(cur_images)
                    index = np.arange(total_data, total_data + len(cur_locs) - 2)
                else:
                    locs = np.concatenate([locs, cur_locs], axis=0)
                    angles = np.concatenate([angles, cur_angles], axis=0)
                    images = np.concatenate([images, cur_images], axis=0)
                    index = np.concatenate([index, np.arange(total_data, total_data + len(cur_locs) - 2)], axis=0)

                total_data += len(cur_locs)
        assert (len(locs) == len(angles)) and (len(locs) == len(images))
        assert (len(locs) == total_data) and (len(locs) == len(index) + 2 * len(seqs))

        tmp_dict[tmp_path] = [index, locs, angles, images]
        all_locs.append(locs)

    all_locs = np.concatenate(all_locs, axis=0)
    min_locs = np.min(all_locs, axis=0)
    print(s, min_locs)
    np.save(os.path.join(scene_path, 'min_loc_all.npy'),  min_locs)

    for tmp_path in tmp_dict.keys():
        index, locs, angles, images = tmp_dict[tmp_path]
        locs -= min_locs
        cur_data = {'index': index, 'locs': locs, 'angles': angles, 'images': images}
        with open(tmp_path, 'wb') as handle:
            pickle.dump(cur_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


