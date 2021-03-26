from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import tensorflow as tf
import time
import os
from utils import *
import transforms3d.euler as txe
import pickle
import math

## This code does data preparation for training the camera pose estimation model (posenet + our camera pose embedding)
## It loads in the pretrained model by novel view synthesis task and uses the model to encode the camera pose for each
## scene. It then save the encoded embedding as well as the decoding system which will can be directly used by the camera pose
## estimation model training.

class encoder(object):
    def __init__(self, sess):
        self.sess = sess
        self.num_block = 4
        self.block_size = 8
        self.grid_loc = 0.1
        self.grid_angle = np.pi / 18
        self.hidden_dim = self.num_block * self.block_size
        self.in_grid_theta = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_theta')
        self.in_rest_theta = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_theta')
        self.in_grid_phi = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_phi')
        self.in_rest_phi = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_phi')
        self.in_grid_ksi = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_ksi')
        self.in_rest_ksi = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_ksi')
        self.in_grid_x = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_x')
        self.in_rest_x = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_x')
        self.in_grid_y = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_y')
        self.in_rest_y = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_y')
        self.in_grid_z = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_z')
        self.in_rest_z = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_z')

        self.v_theta = tf.get_variable('Rotation_v_theta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[36, self.hidden_dim]), dtype=tf.float32))
        self.v_phi = tf.get_variable('Rotation_v_phi', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[18, self.hidden_dim]), dtype=tf.float32))
        self.v_ksi = tf.get_variable('Rotation_v_ksi', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[36, self.hidden_dim]), dtype=tf.float32))
        self.v_x = tf.get_variable('Rotation_v_x', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[40, self.hidden_dim]), dtype=tf.float32))
        self.v_y = tf.get_variable('Rotation_v_y', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[15, self.hidden_dim]), dtype=tf.float32))
        self.v_z = tf.get_variable('Rotation_v_z', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[30, self.hidden_dim]), dtype=tf.float32))

        self.B_theta = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_theta', antisym=True)
        self.B_phi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_phi', antisym=True)
        self.B_ksi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_ksi', antisym=True)
        self.B_x = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_x', antisym=True)
        self.B_y = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_y', antisym=True)
        self.B_z = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size,name='Rotation_B_z', antisym=True)

        # encoding pose to vectors
        self.v_theta_reg = self.v_theta / (tf.norm(self.v_theta, axis=-1, keep_dims=True) + 1e-6)
        self.v_phi_reg = self.v_phi / (tf.norm(self.v_phi, axis=-1, keep_dims=True) + 1e-6)
        self.v_ksi_reg = self.v_ksi / (tf.norm(self.v_ksi, axis=-1, keep_dims=True) + 1e-6)
        self.v_x_reg = self.v_x / (tf.norm(self.v_x, axis=-1, keepdims=True) + 1e-6)
        self.v_y_reg = self.v_y / (tf.norm(self.v_y, axis=-1, keepdims=True) + 1e-6)
        self.v_z_reg = self.v_z / (tf.norm(self.v_z, axis=-1, keepdims=True) + 1e-6)

        self.vec_x = self.get_grid_code(self.v_x_reg, self.B_x, self.in_grid_x, self.in_rest_x, self.grid_loc)
        self.vec_y = self.get_grid_code(self.v_y_reg, self.B_y, self.in_grid_y, self.in_rest_y, self.grid_loc)
        self.vec_z = self.get_grid_code(self.v_z_reg, self.B_z, self.in_grid_z, self.in_rest_z, self.grid_loc)
        self.vec_theta = self.get_grid_code(self.v_theta_reg, self.B_theta, self.in_grid_theta, self.in_rest_theta, self.grid_angle)
        self.vec_phi = self.get_grid_code(self.v_phi_reg, self.B_phi, self.in_grid_phi, self.in_rest_phi, self.grid_angle)
        self.vec_ksi = self.get_grid_code(self.v_ksi_reg, self.B_ksi, self.in_grid_ksi, self.in_rest_ksi, self.grid_angle)

        self.saver = tf.train.Saver()


    def rot(self, grid_code, rest, grid_length, B):
        rest = rest * grid_length
        M = self.get_M(B, rest)
        grid_code = self.motion_model(M, grid_code)
        return grid_code

    def get_grid_code(self, v, B, grid, rest, grid_length):
        grid_code = tf.gather(v, grid, axis=0)
        grid_code = self.rot(grid_code, rest, grid_length, B)
        return grid_code

    def get_M(self, B, a):
        B_re = tf.expand_dims(B, axis=0)
        a_re = tf.reshape(a, [-1, 1, 1])
        M = tf.expand_dims(tf.eye(self.hidden_dim), axis=0) + B_re * a_re + tf.matmul(B_re, B_re) * (a_re ** 2) / 2
        return M

    def motion_model(self, M, grid_code):
        grid_code_new = tf.matmul(M, tf.expand_dims(grid_code, -1))
        grid_code_new = tf.reshape(grid_code_new, [-1, self.hidden_dim])
        return grid_code_new

    def quantize_angle(self, angle):
        grid_angle = np.floor(angle)
        grid_angle[:, 0] = np.clip(grid_angle[:, 0], 0, 35)
        grid_angle[:, 1] = np.clip(grid_angle[:, 1], 0, 17)
        grid_angle[:, 2] = np.clip(grid_angle[:, 2], 0, 35)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def quantize_loc(self, loc):
        grid_loc = np.floor(loc)
        grid_loc[:, 0] = np.clip(grid_loc[:, 0], 0, 39)
        grid_loc[:, 1] = np.clip(grid_loc[:, 1], 0, 14)
        grid_loc[:, 2] = np.clip(grid_loc[:, 2], 0, 29)
        rest_loc = loc - grid_loc.astype(np.float32)
        return grid_loc.astype(np.int32), rest_loc.astype(np.float32)

    def encode(self, loc, angle):
        grid_loc, rest_loc = self.quantize_loc(loc)
        grid_angle, rest_angle = self.quantize_angle(angle)
        feed_dict = {
            self.in_grid_x: grid_loc[:, 0],
            self.in_grid_y: grid_loc[:, 1],
            self.in_grid_z: grid_loc[:, 2],
            self.in_rest_x: rest_loc[:, 0],
            self.in_rest_y: rest_loc[:, 1],
            self.in_rest_z: rest_loc[:, 2],
            self.in_grid_theta: grid_angle[:, 0],
            self.in_grid_phi: grid_angle[:, 1],
            self.in_grid_ksi: grid_angle[:, 2],
            self.in_rest_theta: rest_angle[:, 0],
            self.in_rest_phi: rest_angle[:, 1],
            self.in_rest_ksi: rest_angle[:, 2]
        }
        vec_x, vec_y, vec_z, vec_theta, vec_phi, vec_ksi = self.sess.run([self.vec_x, self.vec_y, self.vec_z, self.vec_theta, self.vec_phi, self.vec_ksi], feed_dict=feed_dict)
        return vec_x, vec_y, vec_z, vec_theta, vec_phi, vec_ksi



    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = '../dataset/7Scenes'
scenes = os.listdir(path)
scenes.sort()
with tf.Session() as sess:

    start_time = time.time()
    my_encoder = encoder(sess)
    my_encoder.load('./checkpoint')

    count = 0
    for s in scenes:
        if '.pickle' in s:
            continue
        print(s)
        scene_path = os.path.join(path, s)
        min_locs = np.squeeze(np.load(os.path.join(scene_path, 'min_loc_all.npy')))
        seqs = [n for n in os.listdir(scene_path) if 'seq' in n]
        for seq in seqs:
            seq_path = os.path.join(scene_path, seq)
            pose_files = [n for n in os.listdir(seq_path) if 'pose.txt' in n]
            for i in range(len(pose_files)):
                f = 'frame-{:06d}.pose.txt'.format(i)
                ps = np.loadtxt(os.path.join(seq_path, f)).flatten()[:12]
                loc = ps[[3, 7, 11]]
                m = ps.reshape((3, 4))[:3, :3]
                theta, phi, gamma = txe.mat2euler(m, 'sxyz')
                loc -= min_locs
                angle = np.array((theta + np.pi, phi + np.pi/2, gamma + np.pi)) #translate to [0, 2pi] [0, pi] [0 2pi] for encoding
                loc = np.reshape(loc, (1, 3)) / 0.1
                angle = np.reshape(angle, (1, 3)) / (np.pi / 18)
                vec_x, vec_y, vec_z, vec_theta, vec_phi, vec_ksi  = my_encoder.encode(loc, angle)
                v_all = np.concatenate([np.squeeze(vec_x), np.squeeze(vec_y), np.squeeze(vec_z), \
                                        np.squeeze(vec_theta), np.squeeze(vec_phi), np.squeeze(vec_ksi)], axis=0)
                np.save(os.path.join(seq_path, 'frame_{:06d}_posvec.npy').format(i), v_all)
                count += 1
                if count % 500 == 0:
                    print('Finish {} time {:.3f}'.format(count, time.time() - start_time))

    # decoding system
    x, y, z = np.arange(0, 40, 0.1), np.arange(0, 15, 0.1), np.arange(0, 30, 0.1)
    theta, phi, ksi = np.arange(0, 36, 0.1), np.arange(0, 18, 0.1), np.arange(0, 36, 0.1)

    grid_x = np.floor(x)
    rest_x = x - grid_x
    grid_y = np.floor(y)
    rest_y = y - grid_y
    grid_z = np.floor(z)
    rest_z = z - grid_z
    grid_theta = np.floor(theta)
    rest_theta = theta - grid_theta
    grid_phi = np.floor(phi)
    rest_phi = phi - grid_phi
    grid_ksi = np.floor(ksi)
    rest_ksi = ksi - grid_ksi

    feed_dict ={
        my_encoder.in_grid_x: grid_x,
        my_encoder.in_grid_y: grid_y,
        my_encoder.in_grid_z: grid_z,
        my_encoder.in_grid_theta: grid_theta,
        my_encoder.in_grid_phi: grid_phi,
        my_encoder.in_grid_ksi: grid_ksi,
        my_encoder.in_rest_x: rest_x,
        my_encoder.in_rest_y: rest_y,
        my_encoder.in_rest_z: rest_z,
        my_encoder.in_rest_theta: rest_theta,
        my_encoder.in_rest_phi: rest_phi,
        my_encoder.in_rest_ksi: rest_ksi
    }

    vec_x, vec_y, vec_z, vec_theta, vec_phi, vec_ksi = sess.run(\
        [my_encoder.vec_x, my_encoder.vec_y, my_encoder.vec_z, my_encoder.vec_theta, my_encoder.vec_phi, my_encoder.vec_ksi], feed_dict=feed_dict)
    x_norms = np.transpose(np.linalg.norm(vec_x, axis=-1, keepdims=True))
    vec_x = np.transpose(vec_x)
    y_norms = np.transpose(np.linalg.norm(vec_y, axis=-1, keepdims=True))
    vec_y = np.transpose(vec_y)
    z_norms = np.transpose(np.linalg.norm(vec_z, axis=-1, keepdims=True))
    vec_z = np.transpose(vec_z)
    theta_norms = np.transpose(np.linalg.norm(vec_theta, axis=-1, keepdims=True))
    vec_theta = np.transpose(vec_theta)
    phi_norms = np.transpose(np.linalg.norm(vec_phi, axis=-1, keepdims=True))
    vec_phi = np.transpose(vec_phi)
    ksi_norms = np.transpose(np.linalg.norm(vec_ksi, axis=-1, keepdims=True))
    vec_ksi = np.transpose(vec_ksi)

    # translate angles back to (-np.pi, np.pi) (-np.pi/2, np.pi/2) (-np.pi, np.pi) to be used in camera pose estimation
    decoding_system = {
        'x': x * 0.1, 'y': y * 0.1, 'z': z * 0.1,\
        'theta': (theta * np.pi / 18 - np.pi), 'phi': (phi * np.pi / 18 - np.pi/2), 'ksi': (ksi * np.pi / 18 - np.pi),\
        'vec_x': vec_x, 'x_norms': x_norms, 'vec_y': vec_y, 'y_norms': y_norms, 'vec_z': vec_z, 'z_norms': z_norms , \
        'vec_theta': vec_theta, 'theta_norms': theta_norms, 'vec_phi': vec_phi, 'phi_norms': phi_norms, 'vec_ksi': vec_ksi, 'ksi_norms': ksi_norms
        }

    with open(os.path.join(path, 'decoding_sys.pickle'), 'wb') as handle:
        pickle.dump(decoding_system, handle, protocol=pickle.HIGHEST_PROTOCOL)




