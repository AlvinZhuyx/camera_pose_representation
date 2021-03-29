from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import pickle
import os
from utils import *
from matplotlib import pyplot as plt
from data_reader import DataReader
from matplotlib import cm
import argparse
import math


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
        self.nsee_per_ins = FLAGS.nsee_per_ins
        self.npred_per_ins = FLAGS.npred_per_ins
        self.nimg_per_ins = self.nsee_per_ins + self.npred_per_ins
        self.hidden_dim = self.num_block * self.block_size
        self.recons_weight = FLAGS.recons_weight
        self.rot_reg_weight = FLAGS.rot_reg_weight
        self.B_reg_weight = FLAGS.B_reg_weight
        self.checkpoint_dir = FLAGS.checkpoint_dir
        self.sample_dir = FLAGS.sample_dir
        self.sess = sess
        self.print_iter = FLAGS.print_iter
        self.grid_angle = FLAGS.grid_angle
        self.grid_num = FLAGS.grid_num
        self.num_B_loc = FLAGS.num_B_loc
        self.num_B_dtheta = FLAGS.num_B_dtheta
        self.location_num = FLAGS.location_num
        self.location_length = FLAGS.location_length
        self.update_step_sz = FLAGS.update_step_sz
        self.path = FLAGS.data_path
        dataset = DataReader(mode='test', dataset='rooms_free_camera_no_object_rotations', context_size=self.nimg_per_ins, root=FLAGS.data_path)
        self.data_loader = dataset.read(batch_size=self.nins_per_batch)

    def encoder(self, in_img, in_pos, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            # 64 * 64 * 3 -> 32 * 32 * 256
            h1 = tf.layers.conv2d(in_img, filters=256, kernel_size=2, strides=2, padding='same')
            h1 = tf.nn.relu(h1)
            # 32 * 32 * 256 -> 32 * 32 * 128
            h21 = tf.layers.conv2d(h1, filters=128, kernel_size=3, strides=1, padding='same')
            h22 = tf.layers.conv2d(h1, filters=128, kernel_size=1, strides=1, padding='same')
            h2 = tf.nn.relu(h21 + h22)
            # 32 * 32 * 128 -> 16 * 16 * (256 + ins_pos)
            h3 = tf.layers.conv2d(h2, filters=256, kernel_size=2, strides=2, padding='same')
            h3 = tf.nn.relu(h3)
            in_pos_tile = tf.tile(tf.reshape(in_pos, [-1, 1, 1, in_pos.shape[-1]]), [1, h3.shape[1], h3.shape[2], 1])
            h3 = tf.concat([h3, in_pos_tile], axis=-1)
            # 16 * 16 * (256 + ins_pos) -> 16 * 16 * 128
            h41 = tf.layers.conv2d(h3, filters=128, kernel_size=3, strides=1, padding='same')
            h42 = tf.layers.conv2d(h3, filters=128, kernel_size=1, strides=1, padding='same')
            h4 = tf.nn.relu(h41 + h42)
            # 16 * 16 * 128 -> 16 * 16 * 256
            h5 = tf.layers.conv2d(h4, filters=256, kernel_size=3, strides=1, padding='same')
            h5 = tf.nn.relu(h5)
            # 16 * 16 * 256 -> 8 * 8 * 128
            h6 = tf.layers.conv2d(h5, filters=256, kernel_size=2, strides=2, padding='same')
            h6 = tf.nn.relu(h6)
            h71 = tf.layers.conv2d(h6, filters=128, kernel_size=3, strides=1, padding='same')
            h72 = tf.layers.conv2d(h6, filters=128, kernel_size=1, strides=1, padding='same')
            h7 = tf.nn.relu(h71 + h72)

            # 8 * 8 * 128 -> 4 * 4 * 128
            h8 = tf.layers.conv2d(h7, filters=256, kernel_size=2, strides=2, padding='same')
            h8 = tf.nn.relu(h8)
            h91 = tf.layers.conv2d(h8, filters=128, kernel_size=3, strides=1, padding='same')
            h92 = tf.layers.conv2d(h8, filters=128, kernel_size=1, strides=1, padding='same')
            h9 = tf.nn.relu(h91 + h92)

            # average pooling
            h10 = tf.layers.conv2d(h9, filters=128, kernel_size=3, strides=1, padding='same')
            ins_vec = tf.layers.average_pooling2d(h10, pool_size=4, strides=4, padding='same')
            gamma8 = tf.layers.conv2d(h7, filters=256, kernel_size=3, strides=1, padding='same')
            gamma8 = tf.layers.average_pooling2d(gamma8, pool_size=8, strides=8, padding='same')
            beta8 = tf.layers.conv2d(h7, filters=256, kernel_size=3, strides=1, padding='same')
            beta8 = tf.layers.average_pooling2d(beta8, pool_size=8, strides=8, padding='same')
            gamma16 = tf.layers.conv2d(h5, filters=256, kernel_size=3, strides=1, padding='same')
            gamma16 = tf.layers.average_pooling2d(gamma16, pool_size=16, strides=16, padding='same')
            beta16 = tf.layers.conv2d(h5, filters=256, kernel_size=3, strides=1, padding='same')
            beta16 = tf.layers.average_pooling2d(beta16, pool_size=16, strides=16, padding='same')
        return ins_vec, gamma8, beta8, gamma16, beta16

    def decoder(self, input, gamma8, beta8, gamma16, beta16, reuse=False):
        def ins_norm(x, gamma, beta):
            assert len(x.shape) == 4
            u = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            sigma = tf.sqrt(tf.reduce_mean(tf.square(x - u), axis=[1, 2], keep_dims=True) + 1e-7)
            gamma = tf.reshape(gamma, (gamma.shape[0], 1, 1, gamma.shape[-1]))
            beta = tf.reshape(beta, (beta.shape[0], 1, 1, beta.shape[-1]))
            norm_x = gamma * (x - u) / sigma + beta
            return norm_x

        with tf.variable_scope('decoder', reuse=reuse):
            # 4
            h0 = tf.layers.dense(input, 4 * 4 * 256, name='h1')
            h0 = tf.reshape(h0, (-1, 4, 4, 256))
            # 8
            h1 = tf.keras.layers.UpSampling2D((2, 2))(h0)
            h21 = tf.layers.conv2d_transpose(tf.nn.leaky_relu(h1), 256, 3, 1, padding='same', name='h21_1')
            h21 = tf.nn.leaky_relu(h21)
            h21 = tf.layers.conv2d_transpose(h21, 256, 3, 1, padding='same', name='h21_2')
            h21 = ins_norm(h21, gamma8, beta8)
            h21 = tf.nn.leaky_relu(h21)
            h22 = tf.layers.conv2d_transpose(h1, 256, 1, 1, padding='same', name='h22')
            h2 = h21 + h22

            # 16
            h3 = tf.keras.layers.UpSampling2D((2, 2))(h2)
            h41 = tf.layers.conv2d_transpose(h3, 256, 3, 1, padding='same', name='h41_1')
            h41 = tf.nn.leaky_relu(h41)
            h41 = tf.layers.conv2d_transpose(h41, 256, 3, 1, padding='same', name='h41_2')
            h41 = ins_norm(h41, gamma16, beta16)
            h41 = tf.nn.leaky_relu(h41)
            h42 = tf.layers.conv2d_transpose(h3, 256, 1, 1, padding='same', name= 'h42')
            h4 = h41 + h42

            # 32
            h5 = tf.keras.layers.UpSampling2D((2, 2))(h4)
            h61 = tf.layers.conv2d_transpose(h5, 128, 3, 1, padding='same', name='h61_1')
            h61 = tf.nn.leaky_relu(h61)
            h61 = tf.layers.conv2d_transpose(h61, 128, 3, 1, padding='same', name='h61_2')
            h61 = tf.nn.leaky_relu(h61)
            h62 = tf.layers.conv2d_transpose(h5, 128, 1, 1, padding='same', name='h62')
            h6 = h61 + h62

            # 64
            h7 = tf.keras.layers.UpSampling2D((2, 2))(h6)
            h81 = tf.layers.conv2d_transpose(h7, 64, 3, 1, padding='same', name='h81_1')
            h81 = tf.nn.leaky_relu(h81)
            h81 = tf.layers.conv2d_transpose(h81, 3, 3, 1, padding='same', name='h81_2')
            h82 = tf.layers.conv2d_transpose(h7, 3, 1, 1, padding='same', name='h82')
            h8 = h81 + h82
        return tf.nn.tanh(h8)

    def build_model(self):
        # currently we seperate the angle changes
        self.in_image_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins, self.im_sz, self.im_sz, self.channel])
        self.in_image_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins, self.im_sz, self.im_sz, self.channel])
        # theta1 and theta0 should be converted into grid space
        self.in_grid_angle_see = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.nsee_per_ins], name='in_asee_grid')
        self.in_rest_angle_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins], name='in_asee_rest')
        self.in_grid_location_see = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.nsee_per_ins, 2], name='in_lsee_grid')
        self.in_rest_location_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins, 2], name='in_lsee_rest')

        self.in_grid_angle_pred = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npred_per_ins], name='in_apred_grid')
        self.in_rest_angle_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins], name='in_apred_rest')
        self.in_grid_location_pred = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npred_per_ins, 2], name='in_lpred_grid')
        self.in_rest_location_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins, 2], name='in_lpred_rest')

        self.add_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None], name='add_a0')
        self.add_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[None], name='add_a1')
        self.add_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None], name='add_a0_rest')
        self.add_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[None], name='add_a1_rest')

        self.add_grid_location0 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='add_l0_grid')
        self.add_grid_location1 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='add_l1_grid')
        self.add_rest_location0 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='add_l0_rest')
        self.add_rest_location1 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='add_l1_rest')


        self.v_location = tf.get_variable('Location_v', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_view = tf.get_variable('Rotation_v', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))

        self.B_rot = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size, name='Rotation_B', antisym=True), [self.hidden_dim, self.hidden_dim])
        self.B_loc = construct_block_diagonal_weights(num_channel=self.num_B_loc, num_block=self.num_block, block_size=self.block_size, name='Location_B', antisym=True)
        self.T_loc = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size, name='Location_T', antisym=True), [self.hidden_dim, self.hidden_dim])


        def get_R(dtheta):
            return tf.eye(self.hidden_dim) + self.T_loc * dtheta + tf.matmul(self.T_loc, self.T_loc) * (dtheta ** 2) / 2.0

        self.R_list = []
        for i in range(self.num_B_dtheta + 1):
            self.R_list.append(get_R(i*2*np.pi / self.num_B_loc))
        self.R_list = tf.stack(self.R_list)

        self.v_view_reg = self.v_view / tf.norm(self.v_view, axis=-1, keep_dims=True)
        v_loc_reshape = tf.reshape(self.v_location, (self.location_num + 1, self.location_num + 1, self.num_block, self.block_size))
        v_loc_reg = v_loc_reshape / (tf.norm(v_loc_reshape, axis=-1, keep_dims=True) * self.num_block)
        self.v_loc_reg = tf.reshape(v_loc_reg, (self.location_num + 1, self.location_num + 1, self.hidden_dim))

        # rotation for head system
        v_view_see = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle_see, self.in_rest_angle_see)
        v_loc_see = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location_see, self.in_rest_location_see)
        v_view_pred = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle_pred, self.in_rest_angle_pred)
        v_loc_pred = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location_pred, self.in_rest_location_pred)

        v_pos_see = tf.concat([v_view_see, v_loc_see], axis=1)
        ins_encode, gamma8, beta8, gamma16, beta16 = self.encoder(self.in_image_see, v_pos_see, reuse=False)

        def inf_ensemble(h_in):
            shape = h_in.shape.as_list()
            assert len(shape) == 4
            assert shape[0] == self.nins_per_batch * self.nsee_per_ins
            h_in = tf.reshape(h_in, (self.nins_per_batch, self.nsee_per_ins, shape[1], shape[2], shape[3]))
            h_in = tf.reduce_sum(h_in, axis=1, keep_dims=True)
            h_in = tf.tile(h_in, (1, self.npred_per_ins, 1, 1, 1))
            h_in = tf.reshape(h_in, (self.nins_per_batch * self.npred_per_ins, shape[1], shape[2], shape[3]))
            return h_in

        ins_encode = tf.squeeze(inf_ensemble(ins_encode))
        gamma8 = inf_ensemble(gamma8)
        beta8 = inf_ensemble(beta8)
        gamma16 = inf_ensemble(gamma16)
        beta16 = inf_ensemble(beta16)
        # reconstruction
        v_total = tf.concat([ins_encode, v_view_pred, v_loc_pred], axis=-1)
        self.re_img = self.decoder(v_total, gamma8, beta8, gamma16, beta16, reuse=False)
        self.recons_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img - self.in_image_pred), axis=0))

        # location rotation loss
        v_view_add0 = self.get_grid_code_rot(self.v_view_reg, self.add_grid_angle0, self.add_rest_angle0)
        v_view_add1 = self.get_grid_code_rot(self.v_view_reg, self.add_grid_angle1, self.add_rest_angle1)
        v_loc_add0 = self.get_grid_code_loc(self.v_loc_reg, self.add_grid_location0, self.add_rest_location0)
        v_loc_add1 = self.get_grid_code_loc(self.v_loc_reg, self.add_grid_location1, self.add_rest_location1)
        loc1_add = (tf.cast(self.add_grid_location1, tf.float32) + self.add_rest_location1) * self.location_length
        loc0_add = (tf.cast(self.add_grid_location0, tf.float32) + self.add_rest_location0) * self.location_length
        delta_location_add = loc1_add - loc0_add
        M_loc_add = self.get_M_loc(self.B_loc, delta_location_add)
        v_loc_add0_rot = self.motion_model(M_loc_add, v_loc_add0)
        
        self.rot_loss_loc = self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_loc_add0_rot - v_loc_add1), axis=0))

        # head system rotation loss
        delta_angle_add = (tf.cast(self.add_grid_angle1, tf.float32) + self.add_rest_angle1 - tf.cast(self.add_grid_angle0, tf.float32) - self.add_rest_angle0) * self.grid_angle
        M_view_add = self.get_M_rot(self.B_rot, delta_angle_add)
        v_view_add0_rot = self.motion_model(M_view_add, v_view_add0)

        self.rot_loss_view = self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_view_add0_rot - v_view_add1), axis=0))

        # B rotation loss
        theta, dtheta = np.meshgrid(np.arange(self.num_B_loc), np.arange(1, self.num_B_dtheta+1))
        theta, dtheta = np.reshape(theta, [-1]), np.reshape(dtheta, [-1])
        theta_prime = (theta + dtheta) % self.num_B_loc
        def RB(R, B):
            return tf.matmul(R, B)
        def B_loc(theta):
            return tf.gather(self.B_loc, theta)
        def R(dtheta):
            return tf.gather(self.R_list, dtheta)
        self.B_rotation_loss = self.B_reg_weight * tf.reduce_sum(tf.reduce_mean((RB(R(dtheta), B_loc(theta)) - B_loc(theta_prime)) ** 2, axis=0)) / len(theta) * 36.0


        rot_var = []
        dec_var = []
        enc_var = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name or 'Location' in var.name:
                print('Rotaion variable: ', var.name)
                rot_var.append(var)
            elif 'encoder' in var.name:
                print('Encoder variable: ', var.name)
                enc_var.append(var)
            else:
                print('Decoder variable: ', var.name)
                dec_var.append(var)

        self.total_rot_loss = self.recons_weight * self.recons_loss + self.rot_loss_loc + self.rot_loss_view + self.B_rotation_loss
        self.update_dec = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.recons_loss, var_list=dec_var)
        self.update_enc = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.recons_loss, var_list=enc_var)
        self.update_rot = tf.train.AdamOptimizer(self.update_step_sz, beta1=self.beta1).minimize(self.total_rot_loss, var_list=rot_var)

    def get_grid_code_rot(self, v, grid_angle, rest_angle):
        assert v.shape == (2 * self.grid_num + 1, self.hidden_dim)
        grid_code = tf.gather(v, grid_angle, axis=0)
        angle = tf.expand_dims(rest_angle, axis=-1) * self.grid_angle
        M = self.get_M_rot(self.B_rot, angle)
        grid_code = self.motion_model(M, grid_code)
        return grid_code

    def get_M_rot(self, B, a):
        B_re = tf.expand_dims(B, axis=0)
        a_re = tf.reshape(a, [-1, 1, 1])
        M = tf.expand_dims(tf.eye(self.hidden_dim), axis=0) + B_re * a_re + tf.matmul(B_re, B_re) * (a_re ** 2) / 2
        return M

    def get_grid_code_loc(self, v, grid_loc, rest_loc):
        assert v.shape == (self.location_num + 1, self.location_num + 1, self.hidden_dim)
        v_reshape = tf.reshape(v, (-1, self.hidden_dim))
        v_idx = grid_loc[:, 0] * (self.location_num + 1) + grid_loc[:, 1]
        grid_code = tf.gather(v_reshape, v_idx, axis=0)
        M = self.get_M_loc(self.B_loc, rest_loc * self.location_length)
        grid_code = self.motion_model(M, grid_code)
        return grid_code

    def dx_to_theta_id_dr(self, dx):
        assert len(dx._shape_as_list()) == 2
        theta = tf.math.atan2(dx[:, 1], dx[:, 0]) % (2 * np.pi)
        theta_id = tf.cast(tf.round(theta / (2 * math.pi / self.num_B_loc)), tf.int32)
        dr = tf.sqrt(tf.reduce_sum(dx ** 2, axis=-1))
        return theta_id, dr

    def get_M_loc(self, B, dx):
        theta_id, dr = self.dx_to_theta_id_dr(dx)
        dr = tf.reshape(dr, [-1, 1, 1])
        B_theta = tf.gather(B, theta_id)
        M = tf.expand_dims(tf.eye(self.hidden_dim), axis=0) + B_theta * dr + tf.matmul(B_theta, B_theta) * (dr ** 2) / 2
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
        assert np.max(angle) <= 2 * self.grid_num and np.max(angle) >= 0
        grid_angle = np.floor(angle) + np.random.randint(low=0, high=2, size=angle.shape)
        grid_angle = np.clip(grid_angle, 0, 2 * self.grid_num)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def quantize_loc(self, loc):
        assert np.max(loc) < self.location_num + 1 and np.min(loc) >= 0
        grid_loc = np.floor(loc) + np.random.randint(low=0, high=2, size=loc.shape)
        grid_loc = np.clip(grid_loc, 0, self.location_num)
        rest_loc = loc - grid_loc.astype(np.float32)
        return grid_loc.astype(np.int), rest_loc.astype(np.float32)

    def test_by_noise_all(self, save_path):
        from room_dataset_tf import RoomDataset, transform_img, sample_batch
        from torch.utils.data import DataLoader
        self.nins_per_batch = 64
        self.npred_per_ins = 1

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.noise_view = tf.placeholder(dtype=tf.float32, shape=[self.hidden_dim])
        self.noise_loc = tf.placeholder(dtype=tf.float32, shape=[self.hidden_dim])
        self.in_image_see = tf.placeholder(dtype=tf.float32, shape=[None, self.im_sz, self.im_sz, self.channel])
        self.in_image_pred = tf.placeholder(dtype=tf.float32, shape=[None, self.im_sz, self.im_sz, self.channel])

        # theta1 and theta0 should be converted into grid space
        self.in_grid_angle_see = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.nsee_per_ins], name='in_asee_grid')
        self.in_rest_angle_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins], name='in_asee_rest')
        self.in_grid_location_see = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.nsee_per_ins, 2], name='in_lsee_grid')
        self.in_rest_location_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins, 2], name='in_lsee_rest')

        self.in_grid_angle_pred = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npred_per_ins], name='in_apred_grid')
        self.in_rest_angle_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins], name='in_apred_rest')
        self.in_grid_location_pred = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npred_per_ins, 2], name='in_lpred_grid')
        self.in_rest_location_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins, 2], name='in_lpred_rest')

        self.v_location = tf.get_variable('Location_v', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_view = tf.get_variable('Rotation_v', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.B_rot = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size, name='Rotation_B', antisym=True), [self.hidden_dim, self.hidden_dim])
        self.B_loc = construct_block_diagonal_weights(num_channel=self.num_B_loc, num_block=self.num_block, block_size=self.block_size, name='Location_B', antisym=True)
        self.T_loc = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size, name='Location_T', antisym=True), [self.hidden_dim, self.hidden_dim])

        def get_R(dtheta):
            return tf.eye(self.hidden_dim) + self.T_loc * dtheta + tf.matmul(self.T_loc, self.T_loc) * (dtheta ** 2) / 2.0

        self.R_list = []
        for i in range(self.num_B_dtheta + 1):
            self.R_list.append(get_R(i * 2 * np.pi / self.num_B_loc))
        self.R_list = tf.stack(self.R_list)

        self.v_view_reg = self.v_view / tf.norm(self.v_view, axis=-1, keep_dims=True)
        v_loc_reshape = tf.reshape(self.v_location, (self.location_num + 1, self.location_num + 1, self.num_block, self.block_size))
        v_loc_reg = v_loc_reshape / (tf.norm(v_loc_reshape, axis=-1, keep_dims=True) * self.num_block)
        self.v_loc_reg = tf.reshape(v_loc_reg, (self.location_num + 1, self.location_num + 1, self.hidden_dim))

        # rotation for head system
        v_view_see = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle_see, self.in_rest_angle_see)
        v_loc_see = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location_see, self.in_rest_location_see)
        v_view_pred = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle_pred, self.in_rest_angle_pred)
        v_loc_pred = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location_pred, self.in_rest_location_pred)

        v_view_noise = v_view_pred + tf.expand_dims(self.noise_view, 0) * tf.random.normal(shape=v_view_pred.shape)
        v_loc_noise = v_loc_pred + tf.expand_dims(self.noise_loc, 0) * tf.random.normal(shape=v_loc_pred.shape)

        v_pos_see = tf.concat([v_view_see, v_loc_see], axis=1)
        ins_encode, gamma8, beta8, gamma16, beta16 = self.encoder(self.in_image_see, v_pos_see, reuse=False)

        def inf_ensemble(h_in):
            shape = h_in.shape.as_list()
            assert len(shape) == 4
            assert shape[0] == self.nins_per_batch * self.nsee_per_ins
            h_in = tf.reshape(h_in, (self.nins_per_batch, self.nsee_per_ins, shape[1], shape[2], shape[3]))
            h_in = tf.reduce_sum(h_in, axis=1, keep_dims=True)
            h_in = tf.tile(h_in, (1, self.npred_per_ins, 1, 1, 1))
            h_in = tf.reshape(h_in, (self.nins_per_batch * self.npred_per_ins, shape[1], shape[2], shape[3]))
            return h_in

        ins_encode = tf.squeeze(inf_ensemble(ins_encode))
        gamma8 = inf_ensemble(gamma8)
        beta8 = inf_ensemble(beta8)
        gamma16 = inf_ensemble(gamma16)
        beta16 = inf_ensemble(beta16)
        # reconstruction
        v_total = tf.concat([ins_encode, v_view_noise, v_loc_noise], axis=-1)
        self.re_img = self.decoder(v_total, gamma8, beta8, gamma16, beta16, reuse=False)
        self.recons_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img - self.in_image_pred), axis=0))

        self.saver = tf.train.Saver(max_to_keep=4)
        self.sess.run(tf.global_variables_initializer())
        could_load, checkpoint_counter = self.load()
        assert could_load

        # calculate the magnitude of each position and each_view
        loc_vec, view_vec = self.sess.run([self.v_loc_reg, self.v_view_reg])
        loc_vec = np.reshape(loc_vec, (-1, self.hidden_dim))
        view_vec = np.reshape(view_vec, (-1, self.hidden_dim))
        loc_std = np.std(loc_vec, axis=0)
        view_std = np.std(view_vec, axis=0)

        test_dataset = RoomDataset(root_dir="../dataset/gqn_room/torch", transform=transform_img)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        start_time = time.time()
        avg_Loss = {}
        avg_psnr = {}
        total_data = 0
        for i in range(21):
            avg_Loss[i] = 0.0
            avg_psnr[i] = 0.0
        test_iter = iter(test_loader)
        count = 0
        while True:
            try:
                x_data_test, v_data_test = next(test_iter)
                x_data_test = x_data_test.squeeze().numpy()
                v_data_test = v_data_test.squeeze().numpy()
            except StopIteration:
                break
            tmp_img_see, tmp_pos_see, tmp_img_pred, tmp_pos_pred = sample_batch(x_data_test, v_data_test, mode="test", seed=0)
            cur_test_num = len(tmp_img_pred)
            total_data += cur_test_num
            if cur_test_num < self.nins_per_batch:
                cur_img_see = np.zeros((self.nins_per_batch, self.nsee_per_ins, self.im_sz, self.im_sz, self.channel))
                cur_img_pred = np.zeros((self.nins_per_batch, self.im_sz, self.im_sz, self.channel))
                cur_pos_see = np.zeros((self.nins_per_batch, self.nsee_per_ins, 5))
                cur_pos_pred = np.zeros((self.nins_per_batch, 5))
                cur_img_see[: cur_test_num] = tmp_img_see
                cur_img_pred[: cur_test_num] = tmp_img_pred
                cur_pos_see[: cur_test_num] = tmp_pos_see
                cur_pos_pred[: cur_test_num] = tmp_pos_pred
            else:
                cur_img_see = tmp_img_see
                cur_img_pred = tmp_img_pred
                cur_pos_see = tmp_pos_see
                cur_pos_pred = tmp_pos_pred

            cur_img_see = np.reshape(cur_img_see, (-1, self.im_sz, self.im_sz, self.channel))
            cur_img_pred = np.reshape(cur_img_pred, (-1, self.im_sz, self.im_sz, self.channel))
            cur_loc_see = np.reshape((cur_pos_see[:, :, :2] + 1.0) / self.location_length, (-1, 2))
            cur_loc_pred = np.reshape((cur_pos_pred[:, :2] + 1.0) / self.location_length, (-1, 2))
            cur_view_see = np.reshape((cur_pos_see[:, :, 3] + 2 * np.pi) / self.grid_angle, (-1))
            cur_view_pred = np.reshape((cur_pos_pred[:, 3] + 2 * np.pi) / self.grid_angle, (-1))

            cur_grid_angle_see, cur_rest_angle_see = self.quantize_angle(cur_view_see)
            cur_grid_angle_pred, cur_rest_angle_pred = self.quantize_angle(cur_view_pred)
            cur_grid_loc_see, cur_rest_loc_see = self.quantize_loc(cur_loc_see)
            cur_grid_loc_pred, cur_rest_loc_pred = self.quantize_loc(cur_loc_pred)


            for i in range(13):
                cur_loc_std = 0.5 / 20.0 * i * loc_std
                cur_view_std = 0.5 / 20.0 * i * view_std
                feed_dict = {self.in_image_see: cur_img_see,
                             self.in_image_pred: cur_img_pred,
                             self.in_grid_angle_see: cur_grid_angle_see,
                             self.in_rest_angle_see: cur_rest_angle_see,
                             self.in_grid_angle_pred: cur_grid_angle_pred,
                             self.in_rest_angle_pred: cur_rest_angle_pred,
                             self.in_grid_location_see: cur_grid_loc_see,
                             self.in_rest_location_see: cur_rest_loc_see,
                             self.in_grid_location_pred: cur_grid_loc_pred,
                             self.in_rest_location_pred: cur_rest_loc_pred,
                             self.noise_view: cur_view_std,
                             self.noise_loc: cur_loc_std}
                re_img = self.sess.run(self.re_img, feed_dict=feed_dict)
                re_img = re_img[:cur_test_num]
                cur_psnr = np.sum(calculate_psnr(re_img, tmp_img_pred))
                cur_loss = np.sum(np.square(re_img - tmp_img_pred))
                avg_Loss[i] += cur_loss
                avg_psnr[i] += cur_psnr
                if count < 10:
                    save_images(tmp_img_pred[:36], os.path.join(save_path, "{}_test_gt.png".format(count)))
                    save_images(re_img[:36], os.path.join(save_path, "{}_test_hat.png".format(count)))

            count += 1
            if count % 100 == 0:
                print(i, count, time.time() - start_time, avg_Loss[0] / total_data, avg_psnr[0] / total_data)

        for i in range(13):
            avg_Loss[i] /= total_data
            avg_psnr[i] /= total_data
            print("noise mag {}, time {}, avg_loss {}, avg_psnr {}".format(0.5 / 20 * i, time.time() - start_time,
                                                                           avg_Loss[i], avg_psnr[i]))

    def train(self):
        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=4)
        self.sess.run(tf.global_variables_initializer())
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
            count = checkpoint_counter
        else:
            print(" [!] Load failed...")
            count = 0
        self.sess_data = tf.train.SingularMonitoredSession()
        start_time = time.time()

        total_param = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            print(variable.name, shape, np.prod(shape.as_list()))
            total_param += np.prod(shape.as_list())
        print('total number of parameter is ', total_param)

        bsz = 4000
        avg_rot_view = 0.0
        avg_rot_loc = 0.0
        avg_B_rot = 0.0
        avg_recons = 0.0
        avg_recons_query = 0.0

        for epoch in range(self.epoch):
            d = self.sess_data.run(self.data_loader)
            query = d.query
            context = query.context
            query_camera = query.query_camera
            query_frame = d.target
            context_camera = context.cameras
            context_frame = context.frames
            context_frame = context_frame * 2.0 - 1.0
            query_frame = query_frame * 2.0 - 1.0

            img_see = []
            img_pred = []
            pos_see = []
            pos_pred = []
            for i in range(self.nins_per_batch):
                idx = np.arange(0, self.nimg_per_ins)
                np.random.shuffle(idx)
                see_idx = idx[0: self.nsee_per_ins]
                pred_idx = idx[self.nsee_per_ins:]
                img_see.append(np.copy(context_frame[i][see_idx]))
                img_pred.append(np.copy(context_frame[i][pred_idx]))
                pos_see.append(np.copy(context_camera[i][see_idx]))
                pos_pred.append(np.copy(context_camera[i][pred_idx]))

            img_see = np.concatenate(img_see, axis=0)
            img_pred = np.concatenate(img_pred, axis=0)
            pos_see = np.concatenate(pos_see, axis=0)
            pos_pred = np.concatenate(pos_pred, axis=0)

            loc_see = (pos_see[:, 0:2] + 1.0) / self.location_length
            loc_pred = (pos_pred[:, 0:2] + 1.0) / self.location_length
            angle_see = (pos_see[:, 3] + 2 * np.pi) / self.grid_angle
            angle_pred = (pos_pred[:, 3] + 2 * np.pi) / self.grid_angle


            add_angle0 = np.random.uniform(low=0.0, high=2 * self.grid_num - 1.0, size=(bsz))
            add_angle1 = add_angle0 + np.random.uniform(low=0.5, high=1.0, size=(bsz))
            add_angle1 = np.clip(add_angle1, 0.0, 2 * self.grid_num)
            add_x0 = np.random.uniform(low=1.0, high=self.location_num - 1.0, size=(bsz, 1))
            add_y0 = np.random.uniform(low=1.0, high=self.location_num - 1.0, size=(bsz, 1))
            tmp_direction = np.random.randint(low=0.0, high=self.num_B_loc + 1, size=(bsz, 1)).astype(np.float32) / self.num_B_loc * 2 * np.pi
            tmp_r = np.random.uniform(0.0, 2.0, size=(bsz, 1))
            add_dx = tmp_r * np.cos(tmp_direction)
            add_dy = tmp_r * np.sin(tmp_direction)
            add_x1 = add_x0 + add_dx
            add_x1 = np.clip(add_x1, 0.0, self.location_num)

            add_y1 = add_y0 + add_dy
            add_y1 = np.clip(add_y1, 0.0, self.location_num)

            add_loc0 = np.concatenate([add_x0, add_y0], axis=-1)
            add_loc1 = np.concatenate([add_x1, add_y1], axis=-1)


            cur_grid_angle_see, cur_rest_angle_see = self.quantize_angle(angle_see)
            cur_grid_angle_pred, cur_rest_angle_pred = self.quantize_angle(angle_pred)
            add_grid_angle0, add_rest_angle0 = self.quantize_angle(add_angle0)
            add_grid_angle1, add_rest_angle1 = self.quantize_angle(add_angle1)
            cur_grid_loc_see, cur_rest_loc_see = self.quantize_loc(loc_see)
            cur_grid_loc_pred, cur_rest_loc_pred = self.quantize_loc(loc_pred)
            add_grid_loc0, add_rest_loc0 = self.quantize_loc(add_loc0)
            add_grid_loc1, add_rest_loc1 = self.quantize_loc(add_loc1)

            feed_dict = {self.in_image_see: img_see,
                         self.in_image_pred: img_pred,
                         self.in_grid_angle_see: cur_grid_angle_see,
                         self.in_rest_angle_see: cur_rest_angle_see,
                         self.in_grid_angle_pred: cur_grid_angle_pred,
                         self.in_rest_angle_pred: cur_rest_angle_pred,
                         self.add_grid_angle0: add_grid_angle0,
                         self.add_rest_angle0: add_rest_angle0,
                         self.add_grid_angle1: add_grid_angle1,
                         self.add_rest_angle1: add_rest_angle1,
                         self.in_grid_location_see: cur_grid_loc_see,
                         self.in_rest_location_see: cur_rest_loc_see,
                         self.in_grid_location_pred: cur_grid_loc_pred,
                         self.in_rest_location_pred: cur_rest_loc_pred,
                         self.add_grid_location0: add_grid_loc0,
                         self.add_grid_location1: add_grid_loc1,
                         self.add_rest_location0: add_rest_loc0,
                         self.add_rest_location1: add_rest_loc1}

            res = self.sess.run([self.update_dec, self.update_enc, self.update_rot, self.rot_loss_loc, self.rot_loss_view, self.recons_loss, self.B_rotation_loss, self.re_img], feed_dict=feed_dict)
            _, _, _, rot_loss_loc, rot_loss_view, recons_loss, B_rot_loss, re_img = res

            query_camera_tile = np.reshape(np.tile(np.expand_dims(query_camera, axis=1), (1, self.npred_per_ins, 1)), (self.nins_per_batch * self.npred_per_ins, 5))
            query_loc = (query_camera_tile[:, 0:2] + 1.0) / self.location_length
            query_angle = (query_camera_tile[:, 3] + 2 * np.pi) / self.grid_angle
            query_grid_angle0,  query_rest_angle0 = self.quantize_angle(query_angle)
            query_grid_loc0, query_rest_loc0 = self.quantize_loc(query_loc)
            feed_dict = {self.in_image_see: img_see,
                         self.in_grid_angle_see: cur_grid_angle_see,
                         self.in_rest_angle_see: cur_rest_angle_see,
                         self.in_grid_angle_pred: query_grid_angle0,
                         self.in_rest_angle_pred: query_rest_angle0,
                         self.in_grid_location_see: cur_grid_loc_see,
                         self.in_rest_location_see: cur_rest_loc_see,
                         self.in_grid_location_pred: query_grid_loc0,
                         self.in_rest_location_pred: query_rest_loc0}
            query_img_recons = self.sess.run(self.re_img, feed_dict=feed_dict)
            query_img_recons = np.reshape(query_img_recons, (self.nins_per_batch, self.npred_per_ins, self.im_sz, self.im_sz, self.channel))
            query_img_recons = np.copy(query_img_recons[:, 0])

            recons_psnr = np.mean(calculate_psnr(re_img, img_pred))
            query_recons_psnr = np.mean(calculate_psnr(query_img_recons, query_frame))
            avg_rot_loc += rot_loss_loc / self.print_iter
            avg_rot_view += rot_loss_view / self.print_iter
            avg_B_rot += B_rot_loss / self.print_iter
            avg_recons += recons_loss / self.print_iter
            avg_recons_query += np.sum(np.mean(np.square(query_img_recons - query_frame), axis=0)) / self.print_iter

            if count % self.print_iter == 0:

                print('Iter {} time {:.3f} rot loc {:.3f} rot view {:.3f} rot B {:4f} recons loss {:.3f} psnr {:.2f} recons query {:.3f} query psnr {:.3f}'.\
                      format(epoch, time.time() - start_time, avg_rot_loc, avg_rot_view, avg_B_rot * 100, avg_recons, recons_psnr, avg_recons_query, query_recons_psnr))
                avg_rot_loc = 0.0
                avg_rot_view = 0.0
                avg_B_rot = 0.0
                avg_recons = 0.0
                avg_recons_query = 0.0


            if count % (self.print_iter * 5) == 0:
                save_images(img_see,  os.path.join(self.sample_dir, '{:07d}_ori_see.png'.format(count)))
                save_images(img_pred, os.path.join(self.sample_dir, '{:07d}_ori_pred.png'.format(count)))
                save_images(re_img, os.path.join(self.sample_dir, '{:07d}_recons.png'.format(count)))
                save_images(query_frame, os.path.join(self.sample_dir, '{:07d}_v_ori.png'.format(count)))
                save_images(query_img_recons, os.path.join(self.sample_dir, '{:07d}_v_recons.png'.format(count)))


            if count % (self.print_iter * 10) == 0 and count != 0: #40
                self.save(count)

            count = count + 1

#########################################  config  #########################################

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--epoch', type=int, default=1000000, help='Number of epochs to train')
parser.add_argument('--nins_per_batch', type=int, default=30, help='Number of different instances in one batch')
parser.add_argument('--nsee_per_ins', type=int, default=6, help='Number of image seen for each scene')
parser.add_argument('--npred_per_ins', type=int, default=3, help='Number of image predict for each scene')
parser.add_argument('--print_iter', type=int, default=5000, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # TODO was 0.003
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')
parser.add_argument('--update_step_sz', type=float, default=1e-2, help='Step size for updating camera pose representation')
parser.add_argument('--train', type=bool, default=False, help='train model or test noise')
# weight of different losses
parser.add_argument('--recons_weight', type=float, default=0.05, help='Reconstruction loss weight')
parser.add_argument('--B_reg_weight', type=float, default=0.8, help='Regulariza the rotation of B') #5
parser.add_argument('--rot_reg_weight', type=float, default=100.0, help='Regularization weight for whether vectors agree with each other') #90
parser.add_argument('--dec_sigma', type=float, default=[0.07], help='Std of Gaussian Kernel')
# structure parameters
parser.add_argument('--num_block', type=int, default=6, help='Number of blocks in the representation')
parser.add_argument('--block_size', type=int, default=16, help='Number of neurons per block')
parser.add_argument('--grid_num', type=int, default=18, help='Number of grid angle')
parser.add_argument('--grid_angle', type=float, default=np.pi/18, help='Size of one angle grid')
parser.add_argument('--num_B_loc', type=int, default=144, help='Number of head rotation')
parser.add_argument('--num_B_dtheta', type=int, default=5, help='Number of dtheta in head rotation regularization')
parser.add_argument('--location_num', type=int, default=20, help='Number of location grids')
parser.add_argument('--location_length', type=float, default=2.0 / 20, help='Length of a grid')
# dataset parameters
parser.add_argument('--im_sz', type=int, default=64, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--data_path', type=str, default='../dataset/gqn_room', help='path for dataset')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='path for saving checkpoint')
parser.add_argument('--sample_dir', type=str, default='sample', help='path for save samples')

FLAGS = parser.parse_args()
def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.set_random_seed(2345)
    np.random.seed(1608)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    with tf.Session() as sess:
        model = recons_model(FLAGS, sess)
        if FLAGS.train:
            model.train()
        else:
            model.test_by_noise_all('img_recons')


if __name__ == '__main__':
    tf.app.run()



