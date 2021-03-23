from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
from utils import *
import transforms3d.euler as txe
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math
import pickle

class dataloader(object):
    def __init__(self, path, nimg_per_ins, train=True):
        scenes = os.listdir(path)
        scenes.sort()
        self.data = {}
        assert nimg_per_ins % 2 == 0
        self.npair_per_ins = nimg_per_ins // 2
        self.train = train
        for s in scenes:
            if '.pickle' in s:
                continue
            scene_path = os.path.join(path, s)
            if train:
                file_name = 'train.pickle'
            else:
                file_name = 'test.pickle'
            with open(os.path.join(scene_path, file_name), 'rb') as handle:
                cur_data = pickle.load(handle)
            self.data[s] = cur_data

    def get_keys(self):
        return list(self.data.keys())

    def get_data(self, idx):
        keys = self.get_keys()
        key = keys[idx]
        delta = 2
        if idx == 6 or idx  == 2:
            delta = 3 
        if self.train:
            cur_data = self.data[key]
            index = cur_data['index']
            chosen_idx0 = np.random.choice(index, self.npair_per_ins, replace=True).astype(np.int)
            chosen_idx1 = chosen_idx0 + delta
            chosen_idx1 = np.clip(chosen_idx1, 0, len(cur_data['images']) - 1).astype(np.int)
            img0 = np.copy(cur_data['images'][chosen_idx0])
            img1 = np.copy(cur_data['images'][chosen_idx1])
            loc0 = np.copy(cur_data['locs'][chosen_idx0])
            loc1 = np.copy(cur_data['locs'][chosen_idx1])
            angles0 = np.copy(cur_data['angles'][chosen_idx0])
            angles1 = np.copy(cur_data['angles'][chosen_idx1])
            return [loc0, loc1], [img0, img1], [angles0, angles1]
        else:
            return np.copy(self.data[key]['locs']), np.copy(self.data[key]['images']), np.copy(self.data[key]['angles'])


# in this model, we split each degree of freedom into one vector
class recons_model(object):
    def __init__(self, FLAGS, sess):
        self.beta1 = FLAGS.beta1
        self.lr = FLAGS.lr
        self.num_block = FLAGS.num_block
        self.block_size = FLAGS.block_size
        self.im_sz = FLAGS.im_sz
        self.channel = FLAGS.channel
        self.epoch = FLAGS.epoch
        self.nins_per_batch = FLAGS.nins_per_batch
        self.nimg_per_ins = FLAGS.nimg_per_ins
        self.hidden_dim = self.num_block * self.block_size
        self.recons_weight = FLAGS.recons_weight
        self.rot_reg_weight = FLAGS.rot_reg_weight
        self.checkpoint_dir = FLAGS.checkpoint_dir
        self.sample_dir = FLAGS.sample_dir
        self.sess = sess
        self.print_iter = FLAGS.print_iter
        self.grid_angle = FLAGS.grid_angle
        self.grid_loc = FLAGS.grid_loc
        self.update_step = FLAGS.update_step
        self.update_step_sz = FLAGS.update_step_sz
        self.path = FLAGS.data_path

    def decoder(self, input, ins_idx, reuse=False):

        def ins_norm(x, i, gamma, beta):
            assert len(x.shape) == 4
            C = x.shape[3]
            u = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            sigma = tf.sqrt(tf.reduce_mean(tf.square(x - u), axis=[1, 2], keep_dims=True) + 1e-7)
            gamma = tf.reshape(gamma, (-1, 1, 1, C))
            beta = tf.reshape(beta, (-1, 1, 1, C))
            norm_x = gamma * (x - u) / sigma + beta
            return norm_x

        with tf.variable_scope('decoder', reuse=reuse):
            gamma2 = tf.get_variable('h2_IN_gamma',
                                     initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 256]), dtype=tf.float32))
            beta2 = tf.get_variable('h2_IN_beta',
                                    initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 256]), dtype=tf.float32))
            gamma4 = tf.get_variable('h4_IN_gamma',
                                     initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 256]), dtype=tf.float32))
            beta4 = tf.get_variable('h4_IN_beta',
                                    initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 256]), dtype=tf.float32))
            gamma6 = tf.get_variable('h6_IN_gamma',
                                     initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 128]), dtype=tf.float32))
            beta6 = tf.get_variable('h6_IN_beta',
                                    initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 128]), dtype=tf.float32))
            gamma8 = tf.get_variable('h8_IN_gamma',
                                     initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 64]), dtype=tf.float32))
            beta8 = tf.get_variable('h8_IN_beta',
                                    initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 64]), dtype=tf.float32))
            h1 = tf.layers.dense(input, 4 * 4 * 256, name='h1')
            h1 = tf.nn.leaky_relu(h1)
            # 4
            h1 = tf.reshape(h1, (-1, 4, 4, 256))
            # 8
            h2 = tf.layers.conv2d_transpose(h1, 256, 4, 2, padding='same', name='h2')
            cur_gamma2 = tf.gather(gamma2, ins_idx)
            cur_beta2 = tf.gather(beta2, ins_idx)
            h2 = ins_norm(h2, 2, cur_gamma2, cur_beta2)
            h2 = tf.nn.leaky_relu(h2)
            # 16
            h3 = tf.keras.layers.UpSampling2D((2, 2))(h2)
            h4 = tf.layers.conv2d_transpose(h3, 256, 4, 1, padding='same', name='h4')
            cur_gamma4 = tf.gather(gamma4, ins_idx)
            cur_beta4 = tf.gather(beta4, ins_idx)
            h4 = ins_norm(h4, 4, cur_gamma4, cur_beta4)
            h4 = tf.nn.leaky_relu(h4) + h3
            # 32
            h5 = tf.keras.layers.UpSampling2D((2, 2))(h4)
            h6 = tf.layers.conv2d_transpose(h5, 128, 4, 1, padding='same', name='h6')
            cur_gamma6 = tf.gather(gamma6, ins_idx)
            cur_beta6 = tf.gather(beta6, ins_idx)
            h6 = ins_norm(h6, 6, cur_gamma6, cur_beta6)
            h6 = tf.nn.leaky_relu(h6)
            # 64
            h7 = tf.keras.layers.UpSampling2D((2, 2))(h6)
            h8 = tf.layers.conv2d_transpose(h7, 64, 4, 1, padding='same', name='h8')
            cur_gamma8 = tf.gather(gamma8, ins_idx)
            cur_beta8 = tf.gather(beta8, ins_idx)
            h8 = ins_norm(h8, 8, cur_gamma8, cur_beta8)
            h8 = tf.nn.leaky_relu(h8)
            # 128
            h9 = tf.keras.layers.UpSampling2D((2, 2))(h8)
            h10 = tf.layers.conv2d_transpose(h9, 3, 4, 1, padding='same', name='h10')
        return tf.nn.tanh(h10)

    def build_model(self):
        # currently we seperate the angle changes
        self.in_image0 = tf.placeholder(dtype=tf.float32, shape=[None, self.im_sz, self.im_sz, self.channel])
        self.in_image1 = tf.placeholder(dtype=tf.float32, shape=[None, self.im_sz, self.im_sz, self.channel])
        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[None])
        # angle and location should be converted into grid space
        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='in_a0_grid')
        self.in_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='in_a1_grid')
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='in_a0_rest')
        self.in_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='in_a1_rest')
        self.add_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='add_a0')
        self.add_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='add_a1')
        self.add_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='add_a0_rest')
        self.add_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='add_a1_rest')

        self.in_grid_loc0 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='in_l0_grid')
        self.in_grid_loc1 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='in_l1_grid')
        self.in_rest_loc0 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='in_l0_rest')
        self.in_rest_loc1 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='in_l1_rest')
        self.add_grid_loc0 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='add_l0')
        self.add_grid_loc1 = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='add_l1')
        self.add_rest_loc0 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='add_l0_rest')
        self.add_rest_loc1 = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='add_l1_rest')

        # define the rotation vectors and matrix
        self.v_instance = tf.get_variable('v_instance', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[7, 96]), dtype=tf.float32))
        self.v_theta = tf.get_variable('Rotation_v_theta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[36, self.hidden_dim]),dtype=tf.float32))
        self.v_phi = tf.get_variable('Rotation_v_phi', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[18, self.hidden_dim]),dtype=tf.float32))
        self.v_ksi = tf.get_variable('Rotation_v_ksi', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[36, self.hidden_dim]),dtype=tf.float32))
        self.v_x = tf.get_variable('Rotation_v_x', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[40, self.hidden_dim]),dtype=tf.float32))
        self.v_y = tf.get_variable('Rotation_v_y', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[15, self.hidden_dim]), dtype=tf.float32))
        self.v_z = tf.get_variable('Rotation_v_z', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[30, self.hidden_dim]), dtype=tf.float32))

        self.B_theta = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_theta', antisym=True)
        self.B_phi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_phi', antisym=True)
        self.B_ksi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_ksi', antisym=True)
        self.B_x = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_x', antisym=True)
        self.B_y = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_y', antisym=True)
        self.B_z = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_z', antisym=True)

        # encoding pose to vectors
        self.v_ins_reg = self.v_instance / (tf.norm(self.v_instance, axis=-1, keep_dims=True) + 1e-6)
        self.v_theta_reg = self.v_theta / (tf.norm(self.v_theta, axis=-1, keep_dims=True) + 1e-6)
        self.v_phi_reg = self.v_phi / (tf.norm(self.v_phi, axis=-1, keep_dims=True) + 1e-6)
        self.v_ksi_reg = self.v_ksi / (tf.norm(self.v_ksi, axis=-1, keep_dims=True) + 1e-6)
        self.v_x_reg = self.v_x / (tf.norm(self.v_x, axis=-1, keepdims=True) + 1e-6)
        self.v_y_reg = self.v_y / (tf.norm(self.v_y, axis=-1, keepdims=True) + 1e-6)
        self.v_z_reg = self.v_z / (tf.norm(self.v_z, axis=-1, keepdims=True) + 1e-6)

        v_theta0, v_phi0, v_ksi0 = self.get_angle_vec(self.in_grid_angle0, self.in_rest_angle0)
        v_theta1, v_phi1, v_ksi1 = self.get_angle_vec(self.in_grid_angle1, self.in_rest_angle1)
        v_theta_add0, v_phi_add0, v_ksi_add0 = self.get_angle_vec(self.add_grid_angle0, self.add_rest_angle0)
        v_theta_add1, v_phi_add1, v_ksi_add1 = self.get_angle_vec(self.add_grid_angle1, self.add_rest_angle1)

        v_x0, v_y0, v_z0 = self.get_loc_vec(self.in_grid_loc0, self.in_rest_loc0)
        v_x1, v_y1, v_z1 = self.get_loc_vec(self.in_grid_loc1, self.in_rest_loc1)
        v_x_add0, v_y_add0, v_z_add0 = self.get_loc_vec(self.add_grid_loc0, self.add_rest_loc0)
        v_x_add1, v_y_add1, v_z_add1 = self.get_loc_vec(self.add_grid_loc1, self.add_rest_loc1)

        v_ins = tf.gather(self.v_ins_reg, self.in_ins_idx)
        v_total0 = tf.concat([v_ins, v_x0, v_y0, v_z0, v_theta0, v_phi0, v_ksi0], axis=-1)
        v_total1 = tf.concat([v_ins, v_x1, v_y1, v_z1, v_theta1, v_phi1, v_ksi1], axis=-1)

        # reconstruction loss
        self.re_img0 = self.decoder(v_total0, self.in_ins_idx, reuse=False)
        self.re_img1 = self.decoder(v_total1, self.in_ins_idx, reuse=True)
        self.recons_loss = 0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img0 - self.in_image0), axis=0)) + \
                           0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img1 - self.in_image1), axis=0))


        # rotation loss
        delta_angle = tf.cast(self.in_grid_angle1, tf.float32) + self.in_rest_angle1 \
                      - tf.cast(self.in_grid_angle0, tf.float32) - self.in_rest_angle0
        delta_angle_add = tf.cast(self.add_grid_angle1, tf.float32) + self.add_rest_angle1 \
                          - tf.cast(self.add_grid_angle0, tf.float32) - self.add_rest_angle0

        delta_loc = tf.cast(self.in_grid_loc1, tf.float32) + self.in_rest_loc1 \
                      - tf.cast(self.in_grid_loc0, tf.float32) - self.in_rest_loc0
        delta_loc_add = tf.cast(self.add_grid_loc1, tf.float32) + self.add_rest_loc1 \
                          - tf.cast(self.add_grid_loc0, tf.float32) - self.add_rest_loc0

        v_x0_rot, v_y0_rot, v_z0_rot = self.rot_loc(v_x0, v_y0, v_z0, delta_loc)
        v_x_add0_rot, v_y_add0_rot, v_z_add0_rot = self.rot_loc(v_x_add0, v_y_add0, v_z_add0, delta_loc_add)
        v_theta0_rot, v_phi0_rot, v_ksi0_rot = self.rot_view(v_theta0, v_phi0, v_ksi0, delta_angle)
        v_theta_add0_rot, v_phi_add0_rot, v_ksi_add0_rot = self.rot_view(v_theta_add0, v_phi_add0, v_ksi_add0, delta_angle_add)


        self.rot_view_loss = 0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_theta0_rot - v_theta1), axis=0)) + \
                             0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_phi0_rot - v_phi1), axis=0)) + \
                             0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_ksi0_rot - v_ksi1), axis=0)) + \
                             0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_theta_add0_rot - v_theta_add1), axis=0)) + \
                             0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_phi_add0_rot - v_phi_add1), axis=0)) + \
                             0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_ksi_add0_rot - v_ksi_add1), axis=0))

        self.rot_loc_loss = 0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_x0_rot - v_x1), axis=0)) + \
                            0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_y0_rot - v_y1), axis=0)) + \
                            0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_z0_rot - v_z1), axis=0)) + \
                            0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_x_add0_rot - v_x_add1), axis=0)) + \
                            0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_y_add0_rot - v_y_add1), axis=0)) + \
                            0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_z_add0_rot - v_z_add1), axis=0))

        rot_var = []
        dec_var = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name:
                print('Rotaion variable: ', var.name)
                rot_var.append(var)
            else:
                print('Decoder variable: ', var.name)
                dec_var.append(var)

        self.update_dec = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.recons_loss, var_list=dec_var)
        self.update_rot = tf.train.AdamOptimizer(self.update_step_sz, beta1=self.beta1).minimize(self.recons_weight * self.recons_loss + self.rot_view_loss + self.rot_loc_loss, var_list=rot_var)

    def get_angle_vec(self, in_grid_angle, in_rest_angle):
        v_theta = self.get_grid_code(self.v_theta_reg, self.B_theta, in_grid_angle[:, 0], in_rest_angle[:, 0], self.grid_angle)
        v_phi = self.get_grid_code(self.v_phi_reg, self.B_phi, in_grid_angle[:, 1], in_rest_angle[:, 1], self.grid_angle)
        v_ksi = self.get_grid_code(self.v_ksi_reg, self.B_ksi, in_grid_angle[:, 2], in_rest_angle[:, 2], self.grid_angle)
        return v_theta, v_phi, v_ksi

    def get_loc_vec(self, in_grid_loc, in_rest_loc):
        v_x = self.get_grid_code(self.v_x_reg, self.B_x, in_grid_loc[:, 0], in_rest_loc[:, 0], self.grid_loc)
        v_y = self.get_grid_code(self.v_y_reg, self.B_y, in_grid_loc[:, 1], in_rest_loc[:, 1], self.grid_loc)
        v_z = self.get_grid_code(self.v_z_reg, self.B_z, in_grid_loc[:, 2], in_rest_loc[:, 2], self.grid_loc)
        return v_x, v_y, v_z

    def rot_view(self, v_theta, v_phi, v_ksi, delta_angle):
        v_theta = self.rot(v_theta, delta_angle[:, 0], self.grid_angle, self.B_theta)
        v_phi = self.rot(v_phi, delta_angle[:, 1] , self.grid_angle, self.B_phi)
        v_ksi = self.rot(v_ksi, delta_angle[:, 2], self.grid_angle, self.B_ksi)
        return v_theta, v_phi, v_ksi

    def rot_loc(self, v_x, v_y, v_z, delta_loc):
        v_x = self.rot(v_x, delta_loc[:, 0], self.grid_loc, self.B_x)
        v_y = self.rot(v_y, delta_loc[:, 1], self.grid_loc, self.B_y)
        v_z = self.rot(v_z, delta_loc[:, 2], self.grid_loc, self.B_z)
        return v_x, v_y, v_z

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


    def save(self, step):
        model_name = '3D.model'
        checkpoint_dir = self.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def quantize_angle(self, angle):
        grid_angle = np.floor(angle) + np.random.randint(low=0, high=2, size=angle.shape)
        grid_angle[:, 0] = np.clip(grid_angle[:, 0], 0, 35)
        grid_angle[:, 1] = np.clip(grid_angle[:, 1], 0, 17)
        grid_angle[:, 2] = np.clip(grid_angle[:, 2], 0, 35)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def quantize_loc(self, loc):
        grid_loc = np.floor(loc) + np.random.randint(low=0, high=2, size=loc.shape)
        grid_loc[:, 0] = np.clip(grid_loc[:, 0], 0, 39)
        grid_loc[:, 1] = np.clip(grid_loc[:, 1], 0, 14)
        grid_loc[:, 2] = np.clip(grid_loc[:, 2], 0, 29)
        rest_loc = loc - grid_loc.astype(np.float32)
        return grid_loc.astype(np.int32), rest_loc.astype(np.float32)

    def gen_point(self, pmax, bsz):
        p0 = np.random.uniform(low=0.0, high=pmax - 1, size=(bsz, 1))
        p1 = p0 + np.random.uniform(low=0.5, high=1.0, size=(bsz, 1))
        p1 = np.clip(p1, 0.0, pmax)
        return p0, p1

    def print_loc(self, save_path='v_vis'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(tf.global_variables_initializer())
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        res = self.sess.run([self.v_x_reg, self.v_y_reg, self.v_z_reg, self.v_theta_reg, self.v_phi_reg, self.v_ksi_reg])
        name = ['x', 'y', 'z', 'theta', 'phi', 'ksi']
        for i in range(6):
            v = res[i]
            cor = np.matmul(v, np.transpose(v))
            plt.figure()
            plt.imshow(cor)
            plt.savefig(os.path.join(save_path, name[i] + '.png'))
            plt.close()
            
    def train(self):
        self.train_loader = dataloader(FLAGS.data_path, nimg_per_ins=self.nimg_per_ins, train=True)
        self.test_loader = dataloader(FLAGS.data_path, nimg_per_ins=self.nimg_per_ins, train=False)
        print('--------------------------------Finish loading data-------------------------------')
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
            count = checkpoint_counter
        else:
            print(" [!] Load failed...")
            count = 0

        start_time = time.time()
        avg_rot_view_loss = 0.0
        avg_rot_loc_loss = 0.0
        avg_recons_loss = 0.0

        add_bsz = 3000
        for epoch in range(count, self.epoch):
            cur_idx = []
            cur_img0 = []
            cur_img1 = []
            cur_angle0 = []
            cur_angle1 = []
            cur_loc0 = []
            cur_loc1 = []
            for i in range(7):
                locs, imgs, angles = self.train_loader.get_data(i)
                cur_loc0.append(locs[0])
                cur_loc1.append(locs[1])
                cur_img0.append(imgs[0])
                cur_img1.append(imgs[1])
                cur_angle0.append(angles[0])
                cur_angle1.append(angles[1])
                cur_idx.append(np.array([i] * len(locs[0]), dtype=np.int))

            cur_img0 = np.concatenate(cur_img0, axis=0)
            cur_img1 = np.concatenate(cur_img1, axis=0)
            cur_angle0 = np.concatenate(cur_angle0, axis=0) / self.grid_angle
            cur_angle1 = np.concatenate(cur_angle1, axis=0) / self.grid_angle
            cur_loc0 = np.concatenate(cur_loc0, axis=0) / self.grid_loc
            cur_loc1 = np.concatenate(cur_loc1, axis=0) / self.grid_loc
            cur_idx = np.concatenate(cur_idx, axis=0)

            add_theta0, add_theta1 = self.gen_point(36, add_bsz)
            add_phi0, add_phi1 = self.gen_point(18, add_bsz)
            add_ksi0, add_ksi1 = self.gen_point(36, add_bsz)
            add_x0, add_x1 = self.gen_point(40, add_bsz)
            add_y0, add_y1 = self.gen_point(15, add_bsz)
            add_z0, add_z1 = self.gen_point(30, add_bsz)
            add_angle0 = np.concatenate([add_theta0, add_phi0, add_ksi0], axis=-1)
            add_angle1 = np.concatenate([add_theta1, add_phi1, add_ksi1], axis=-1)
            add_loc0 = np.concatenate([add_x0, add_y0, add_z0], axis=-1)
            add_loc1 = np.concatenate([add_x1, add_y1, add_z1], axis=-1)

            cur_grid_angle0, cur_rest_angle0 = self.quantize_angle(cur_angle0)
            cur_grid_angle1, cur_rest_angle1 = self.quantize_angle(cur_angle1)
            add_grid_angle0, add_rest_angle0 = self.quantize_angle(add_angle0)
            add_grid_angle1, add_rest_angle1 = self.quantize_angle(add_angle1)
            cur_grid_loc0, cur_rest_loc0 = self.quantize_loc(cur_loc0)
            cur_grid_loc1, cur_rest_loc1 = self.quantize_loc(cur_loc1)
            add_grid_loc0, add_rest_loc0 = self.quantize_loc(add_loc0)
            add_grid_loc1, add_rest_loc1 = self.quantize_loc(add_loc1)

            feed_dict = {self.in_image0: cur_img0,
                         self.in_image1: cur_img1,
                         self.in_ins_idx: cur_idx,
                         self.in_grid_angle0: cur_grid_angle0,
                         self.in_rest_angle0: cur_rest_angle0,
                         self.in_grid_angle1: cur_grid_angle1,
                         self.in_rest_angle1: cur_rest_angle1,
                         self.add_grid_angle0: add_grid_angle0,
                         self.add_rest_angle0: add_rest_angle0,
                         self.add_grid_angle1: add_grid_angle1,
                         self.add_rest_angle1: add_rest_angle1,
                         self.in_grid_loc0: cur_grid_loc0,
                         self.in_rest_loc0: cur_rest_loc0,
                         self.in_grid_loc1: cur_grid_loc1,
                         self.in_rest_loc1: cur_rest_loc1,
                         self.add_grid_loc0: add_grid_loc0,
                         self.add_rest_loc0: add_rest_loc0,
                         self.add_grid_loc1: add_grid_loc1,
                         self.add_rest_loc1: add_rest_loc1}

            for _ in range(self.update_step):
                self.sess.run(self.update_rot, feed_dict=feed_dict)

            res = self.sess.run([self.update_dec, self.rot_loc_loss, self.rot_view_loss,  self.recons_loss, self.re_img0, self.re_img1], feed_dict=feed_dict)
            _, rot_loc_loss, rot_view_loss, recons_loss, re_img0, re_img1 = res
            re_img = np.concatenate([np.expand_dims(re_img0, axis=1), np.expand_dims(re_img1, axis=1)], axis=1)
            re_img = np.reshape(re_img, (-1, self.im_sz, self.im_sz, self.channel))
            cur_img = np.concatenate([np.expand_dims(cur_img0, axis=1), np.expand_dims(cur_img1, axis=1)], axis=1)
            cur_img = np.reshape(cur_img, (-1, self.im_sz, self.im_sz, self.channel))
            recons_psnr = np.mean(calculate_psnr(re_img, cur_img))
            avg_rot_loc_loss += rot_loc_loss / self.print_iter
            avg_rot_view_loss += rot_view_loss / self.print_iter
            avg_recons_loss += recons_loss / self.print_iter

            if count % self.print_iter == 0:
                print('Epoch {} time {:.3f} rot loc loss {:.3f} rot view loss {:.3f} recons loss {:.3f} psnr {:.2f}'. \
                    format(epoch, time.time() - start_time, avg_rot_loc_loss, avg_rot_view_loss, avg_recons_loss, recons_psnr))
                avg_rot_loc_loss = 0.0
                avg_rot_view_loss = 0.0
                avg_recons_loss = 0.0

            if count % (self.print_iter * 4) == 0:
                save_images(cur_img0, os.path.join(self.sample_dir, '{:06d}_ori.png'.format(count)))
                save_images(re_img0, os.path.join(self.sample_dir, '{:06d}_rec.png'.format(count)))

            if count % (self.print_iter * 4) == 0 or count == self.epoch:
                self.save(count)

            count = count + 1

#########################################  config  #########################################
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--epoch', type=int, default=100001, help='Number of epochs to train')
parser.add_argument('--nins_per_batch', type=int, default=7, help='Number of different instances in one batch')
parser.add_argument('--nimg_per_ins', type=int, default=32, help='Number of image chosen for one instance in one batch, since we use pair of data, should be multiple of 2')
parser.add_argument('--print_iter', type=int, default=1000, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # TODO was 0.003
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')
parser.add_argument('--update_step', type=int, default=2, help='Number of inference step in Langevin')
parser.add_argument('--update_step_sz', type=float, default=1e-3, help='Step size for Langevin update')
# weight of different losses
parser.add_argument('--recons_weight', type=float, default=0.009, help='Reconstruction loss weight')
parser.add_argument('--rot_reg_weight', type=float, default=50.0, help='Regularization weight for whether vectors agree with each other')
# structure parameters
parser.add_argument('--num_block', type=int, default=4, help='Number of blocks in the representation')
parser.add_argument('--block_size', type=int, default=8, help='Number of neurons per block')
parser.add_argument('--grid_angle', type=float, default=np.pi/18, help='Size of one angle grid')
parser.add_argument('--grid_loc', type=float, default=0.1, help='Size of one angle grid')
# dataset parameters
parser.add_argument('--im_sz', type=int, default=128, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--data_path', type=str, default='../dataset/7Scenes', help='path for dataset')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='path for saving checkpoint')
parser.add_argument('--sample_dir', type=str, default='sample', help='path for save samples')

FLAGS = parser.parse_args()

def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.set_random_seed(1234)
    np.random.seed(42)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    with tf.Session() as sess:
        model = recons_model(FLAGS, sess)
        model.train()



if __name__ == '__main__':
    tf.app.run()




