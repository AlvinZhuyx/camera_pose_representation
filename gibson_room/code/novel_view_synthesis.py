from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
from utils import *
from matplotlib import pyplot as plt
import argparse
import math
import pickle

#########################################  data  #########################################
# a dataloader that load part of the data at one time
class dataloader(object):
    def __init__(self, path, im_sz):
        instances = os.listdir(path)
        instances.sort()

        self.scenes = []
        self.im_sz = im_sz
        for ins in instances:
            ins_path = os.path.join(path, ins)
            items = os.listdir(ins_path)
            for it in items:
                self.scenes.append(os.path.join(ins_path, it))

    def get_num_ins(self):
        return len(self.scenes)

    def get_item(self, idx, num_pair, mode='train', use_all=False):
        # During training, we sample image in pairs
        # Images in each pair has the pose that are close to each other, so that we can apply the camera poses of the two images to the rotation loss
        # This can further help us enforce the rotation consistency in our learned pose representation system

        s = self.scenes[idx]
        if mode == 'train':
            meta_file = os.path.join(s, 'train_meta.json')
        else:
            meta_file = os.path.join(s, 'test_meta.json')

        with open(meta_file, "rb") as f:
            data_dict = pickle.load(f)
            keys = list(data_dict.keys())
        if use_all:
            chosen_keys = data_dict.keys()
        elif mode == 'train':
            chosen_idx = np.random.choice(len(keys), num_pair, replace=False)
            chosen_keys = []
            for idx in chosen_idx:
                cur_key0 = keys[idx]
                chosen_keys.append(cur_key0)
                gx, gy, gtheta = cur_key0
                tmpc = 0
                while True:
                    dx = np.random.randint(low=-1, high=2)
                    dy = np.random.randint(low=-1, high=2)
                    dtheta = np.random.randint(low=-2, high=3)
                    if dx == 0 and dy == 0 and dtheta == 0:
                        continue
                    ngx = gx + dx
                    ngy = gy + dy
                    ngtheta = gtheta + dtheta
                    tmpc += 1
                    assert tmpc < 10000
                    if (ngx, ngy, ngtheta) in keys:
                        chosen_keys.append((ngx, ngy, ngtheta))
                        break
        else:
            chosen_idx = np.random.choice(len(keys), num_pair, replace=False)
            chosen_keys = []
            for idx in chosen_idx:
                chosen_keys.append(keys[idx])

        img_list = []
        pos_list = []
        for key in chosen_keys:
            if mode == 'train':
                file, pos = data_dict[key]
            else:
                points = data_dict[key]
                pidx = np.random.choice(len(points))
                file, pos = points[pidx]

            image_path = os.path.join(s, file)
            img = load_rgb(image_path, self.im_sz)
            img_list.append(img)
            pos_list.append(pos)
        img_list = np.array(img_list)
        pos_list = np.array(pos_list)

        return img_list, pos_list

    def get_all_test(self, idx):
        # load in all the testing data at one time
        s = self.scenes[idx]
        meta_file = os.path.join(s, 'test_meta.json')
        with open(meta_file, "rb") as f:
            data_dict = pickle.load(f)
            keys = list(data_dict.keys())

        img_list = []
        pos_list = []
        for key in keys:
            for file, pos in data_dict[key]:
                image_path = os.path.join(s, file)
                img = load_rgb(image_path, self.im_sz)
                img_list.append(img)
                pos_list.append(pos)

        img_list = np.array(img_list)
        pos_list = np.array(pos_list)

        return img_list, pos_list


######################################### model ##########################################
class decoder_model(object):
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
        self.update_step = FLAGS.update_step
        self.update_step_sz = FLAGS.update_step_sz
        self.path = FLAGS.data_path

        assert self.nimg_per_ins % 2 == 0
        self.npair_per_ins = self.nimg_per_ins // 2
        self.data_loader = dataloader(FLAGS.data_path, im_sz=self.im_sz)

    def decoder(self, input, ins_idx, reuse=False):
        def ins_norm(x, gamma, beta):
            assert len(x.shape) == 4
            C = x.shape[3]
            u = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            sigma = tf.sqrt(tf.reduce_mean(tf.square(x - u), axis=[1, 2], keep_dims=True) + 1e-7)
            gamma = tf.reshape(gamma, (-1, 1, 1, C))
            beta = tf.reshape(beta, (-1, 1, 1, C))
            norm_x = gamma * (x - u) / sigma + beta
            return norm_x

        with tf.variable_scope('decoder', reuse=reuse):
            h1 = tf.layers.dense(input, 4 * 4 * 256, name='h1')
            h1 = tf.nn.leaky_relu(h1)
            # 4
            h1 = tf.reshape(h1, (-1, 4, 4, 256))
            # 8
            h2 = tf.layers.conv2d_transpose(h1, 256, 4, 2, padding='same', name='h2')
            gamma2 = tf.gather(self.gamma2, ins_idx)
            beta2 = tf.gather(self.beta2, ins_idx)
            h2 = ins_norm(h2, gamma2, beta2)
            h2 = tf.nn.leaky_relu(h2)
            # 16
            h3 = tf.keras.layers.UpSampling2D((2, 2))(h2)
            h4 = tf.layers.conv2d_transpose(h3, 256, 4, 1, padding='same', name='h4')
            gamma4 = tf.gather(self.gamma4, ins_idx)
            beta4 = tf.gather(self.beta4, ins_idx)
            h4 = ins_norm(h4, gamma4, beta4)
            h4 = tf.nn.leaky_relu(h4) + h3
            # 32
            h5 = tf.keras.layers.UpSampling2D((2, 2))(h4)
            h6 = tf.layers.conv2d_transpose(h5, 128, 4, 1, padding='same', name='h6')
            gamma6 = tf.gather(self.gamma6, ins_idx)
            beta6 = tf.gather(self.beta6, ins_idx)
            h6 = ins_norm(h6, gamma6, beta6)
            h6 = tf.nn.leaky_relu(h6)
            # 64
            h7 = tf.keras.layers.UpSampling2D((2, 2))(h6)
            h8 = tf.layers.conv2d_transpose(h7, 64, 4, 1, padding='same', name='h8')
            gamma8 = tf.gather(self.gamma8, ins_idx)
            beta8 = tf.gather(self.beta8, ins_idx)
            h8 = ins_norm(h8, gamma8, beta8)
            h8 = tf.nn.leaky_relu(h8)
            # 128
            h9 = tf.keras.layers.UpSampling2D((2, 2))(h8)
            h10 = tf.layers.conv2d_transpose(h9, 3, 4, 1, padding='same', name='h10')

        return tf.nn.tanh(h10)

    def build_model(self):
        # in_image 0 & 1, in_grid_angle 0 & 1, in_rest_angle 0 & 1, in_grid_location 0 & 1, in_rest_location 0 & 1 are the image
        # and pose information for the current visual input. We use pair of images that have close pose so that we can apply the 
		# camera poses of the two images to the rotation loss to help us further enforce the pose rotation loss. 
		# The grid_angle, grid_location indicate the grid point we use to get our pose representation
        # and rest_angle, rest_location denotes the distance we need to rotate from the grid point
        self.in_image0 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, self.im_sz, self.im_sz, self.channel])
        self.in_image1 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, self.im_sz, self.im_sz, self.channel])
        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins])
        # theta1 and theta0 should be converted into grid space
        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins], name='in_a0_grid')
        self.in_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins], name='in_a1_grid')
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins], name='in_a0_rest')
        self.in_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins], name='in_a1_rest')

        self.in_grid_location0 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_l0_grid')
        self.in_grid_location1 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_l1_grid')
        self.in_rest_location0 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_l0_rest')
        self.in_rest_location1 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_l1_rest')

        # add_grid 0 & 1 add_rest 0 & 1 denote the additional pair of pose data we sampled to enforce the pose rotation loss
        self.add_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None], name='add_a0')
        self.add_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[None], name='add_a1')
        self.add_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None], name='add_a0_rest')
        self.add_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[None], name='add_a1_rest')

        self.add_grid_location0 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='add_l0_grid')
        self.add_grid_location1 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='add_l1_grid')
        self.add_rest_location0 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='add_l0_rest')
        self.add_rest_location1 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='add_l1_rest')


        num_ins = self.data_loader.get_num_ins()
        # v_instance and self.gamma2, self.beta2 self.gamma4, self.beta4, self.gamma6, self.beta6, self.gamma8, self.beta8
        # are scene represention vectors that needs to be passed to the decoder (please check the decoder method for their usage)
        # v_location and v_view is the grid points pose representation system where B_rot, B_loc are matrices to rotate
        # T_loc is the matrix to rotate B (i.e the matrix C in our paper)
        self.v_instance = tf.get_variable('v_instance', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 512 + 256]), dtype=tf.float32))
        self.v_location = tf.get_variable('Location_v', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_view = tf.get_variable('Rotation_v', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))

        self.B_rot = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size, name='Rotation_B', antisym=True), [self.hidden_dim, self.hidden_dim])
        self.B_loc = construct_block_diagonal_weights(num_channel=self.num_B_loc, num_block=self.num_block, block_size=self.block_size, name='Location_B', antisym=True)
        self.T_loc = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size, name='Location_T', antisym=True), [self.hidden_dim, self.hidden_dim])

        self.gamma2 = tf.get_variable('h2_IN_gamma', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.beta2 = tf.get_variable('h2_IN_beta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.gamma4 = tf.get_variable('h4_IN_gamma', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.beta4 = tf.get_variable('h4_IN_beta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.gamma6 = tf.get_variable('h6_IN_gamma', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
        self.beta6 = tf.get_variable('h6_IN_beta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
        self.gamma8 = tf.get_variable('h8_IN_gamma', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 64]), dtype=tf.float32))
        self.beta8 = tf.get_variable('h8_IN_beta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 64]), dtype=tf.float32))


        # B rotation loss in the Ploar coordinate system
        def get_R(dtheta):
            return tf.eye(self.hidden_dim) + self.T_loc * dtheta + tf.matmul(self.T_loc, self.T_loc) * (dtheta ** 2) / 2.0
        self.R_list = []
        for i in range(self.num_B_dtheta + 1):
            self.R_list.append(get_R(i*2*np.pi / self.num_B_loc))
        self.R_list = tf.stack(self.R_list)
        theta, dtheta = np.meshgrid(np.arange(self.num_B_loc), np.arange(1, self.num_B_dtheta + 1))
        theta, dtheta = np.reshape(theta, [-1]), np.reshape(dtheta, [-1])
        theta_prime = (theta + dtheta) % self.num_B_loc
        def RB(R, B):
            return tf.matmul(R, B)
        def B_loc(theta):
            return tf.gather(self.B_loc, theta)
        def R(dtheta):
            return tf.gather(self.R_list, dtheta)
        self.B_rotation_loss = tf.reduce_sum(
            tf.reduce_mean((RB(R(dtheta), B_loc(theta)) - B_loc(theta_prime)) ** 2, axis=0)) / len(theta) * 36.0

        # normalize our pose vectors, the location vectors (vector for position) are normalized by each block while the
        # view vectors (vector for orientation) are normalized directly
        self.v_ins_reg = self.v_instance / tf.norm(self.v_instance, axis=-1, keep_dims=True)
        self.v_view_reg = self.v_view / tf.norm(self.v_view, axis=-1, keep_dims=True)
        v_loc_reshape = tf.reshape(self.v_location, (self.location_num + 1, self.location_num + 1, self.num_block, self.block_size))
        v_loc_reg = v_loc_reshape / (tf.norm(v_loc_reshape, axis=-1, keep_dims=True) * self.num_block)
        self.v_loc_reg = tf.reshape(v_loc_reg, (self.location_num + 1, self.location_num + 1, self.hidden_dim))

        # rotation for orientation system
        v_view0 = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle0, self.in_rest_angle0)
        v_view1 = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle1, self.in_rest_angle1)
        v_view_add0 = self.get_grid_code_rot(self.v_view_reg, self.add_grid_angle0, self.add_rest_angle0)
        v_view_add1 = self.get_grid_code_rot(self.v_view_reg, self.add_grid_angle1, self.add_rest_angle1)

        # rotation for location (position) system
        v_loc0 = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location0, self.in_rest_location0)
        v_loc1 = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location1, self.in_rest_location1)
        v_loc_add0 = self.get_grid_code_loc(self.v_loc_reg, self.add_grid_location0, self.add_rest_location0)
        v_loc_add1 = self.get_grid_code_loc(self.v_loc_reg, self.add_grid_location1, self.add_rest_location1)

        # reconstruction loss
        v_ins = tf.gather(self.v_ins_reg, self.in_ins_idx)
        v_total0 = tf.concat([v_ins, v_view0, v_loc0], axis=-1)
        v_total1 = tf.concat([v_ins, v_view1, v_loc1], axis=-1)
        self.re_img0 = self.decoder(v_total0, self.in_ins_idx, reuse=False)
        self.re_img1 = self.decoder(v_total1, self.in_ins_idx, reuse=True)
        self.recons_loss = 0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img0 - self.in_image0), axis=0)) +\
                           0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img1 - self.in_image1), axis=0))

        # location (position) rotation loss
        loc1 = (tf.cast(self.in_grid_location1, tf.float32) + self.in_rest_location1) * self.location_length
        loc0 = (tf.cast(self.in_grid_location0, tf.float32) + self.in_rest_location0) * self.location_length
        loc1_add = (tf.cast(self.add_grid_location1, tf.float32) + self.add_rest_location1) * self.location_length
        loc0_add = (tf.cast(self.add_grid_location0, tf.float32) + self.add_rest_location0) * self.location_length
        delta_location = loc1 - loc0
        delta_location_add = loc1_add - loc0_add
        M_loc = self.get_M_loc(self.B_loc, delta_location)
        v_loc0_rot = self.motion_model(M_loc, v_loc0)
        M_loc_add = self.get_M_loc(self.B_loc, delta_location_add)
        v_loc_add0_rot = self.motion_model(M_loc_add, v_loc_add0)
        
        self.rot_loss_loc = 0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(v_loc0_rot - v_loc1), axis=0)) + \
                        0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(v_loc_add0_rot - v_loc_add1), axis=0))

        # orientation system rotation loss
        delta_angle = (tf.cast(self.in_grid_angle1, tf.float32) + self.in_rest_angle1 - tf.cast(self.in_grid_angle0, tf.float32) - self.in_rest_angle0) * self.grid_angle
        M_view = self.get_M_rot(self.B_rot, delta_angle)
        v_view0_rot = self.motion_model(M_view, v_view0)

        delta_angle_add = (tf.cast(self.add_grid_angle1, tf.float32) + self.add_rest_angle1 - tf.cast(self.add_grid_angle0, tf.float32) - self.add_rest_angle0) * self.grid_angle
        M_view_add = self.get_M_rot(self.B_rot, delta_angle_add)
        v_view_add0_rot = self.motion_model(M_view_add, v_view_add0)

        self.rot_loss_view = 0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(v_view0_rot - v_view1), axis=0)) + \
                             0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(v_view_add0_rot - v_view_add1), axis=0))

        rot_var = []
        dec_var = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name or 'Location' in var.name:
                print('Rotaion variable: ', var.name)
                rot_var.append(var)
            else:
                print('Decoder variable: ', var.name)
                dec_var.append(var)
        
        
        self.total_rot_loss = self.recons_weight * self.recons_loss + self.rot_reg_weight * self.rot_loss_loc \
                              + self.rot_reg_weight * self.rot_loss_view + self.B_reg_weight * self.B_rotation_loss

        self.update_dec = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.recons_loss, var_list=dec_var)
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

    def quantize_angle(self, angle, train=True):
        assert np.max(angle) <= 2 * self.grid_num and np.max(angle) >= 0
        # during traing, we randomly choose upper or lower grid point to get the pose vector
        # to further enforce consistency of our underlying rotation system
        if train:
            grid_angle = np.floor(angle) + np.random.randint(low=0, high=2, size=angle.shape)
        else:
            grid_angle = np.around(angle)
        grid_angle = np.clip(grid_angle, 0, 2 * self.grid_num)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def quantize_loc(self, loc, train=True):
        assert np.max(loc) < self.location_num + 1 and np.min(loc) >= 0
        # during traing, we randomly choose upper or lower grid point to get the pose vector
        # to further enforce consistency of our underlying rotation system
        if train:
            grid_loc = np.floor(loc) + np.random.randint(low=0, high=2, size=loc.shape)
        else:
            grid_loc = np.around(loc)
        grid_loc = np.clip(grid_loc, 0, self.location_num)
        rest_loc = loc - grid_loc.astype(np.float32)
        return grid_loc.astype(np.int), rest_loc.astype(np.float32)

    def test_by_noise(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        num_ins = self.data_loader.get_num_ins()
        self.noise_view = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_dim])
        self.noise_loc = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_dim])

        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[None])
        # theta1 and theta0 should be converted into grid space
        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None], name='in_a0_grid')
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None], name='in_a0_rest')
        self.in_grid_location0 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='in_l0_grid')
        self.in_rest_location0 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='in_l0_rest')

        self.v_instance = tf.get_variable('v_instance', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 768]), dtype=tf.float32))
        self.v_location = tf.get_variable('Location_v', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]),
            dtype=tf.float32))
        # self.u_location = tf.get_variable('Location_u', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_view = tf.get_variable('Rotation_v', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))

        self.B_rot = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size,
                                             name='Rotation_B', antisym=True), [self.hidden_dim, self.hidden_dim])
        self.B_loc = construct_block_diagonal_weights(num_channel=self.num_B_loc, num_block=self.num_block,
                                                      block_size=self.block_size, name='Location_B', antisym=True)
        self.T_loc = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block, block_size=self.block_size,
                                             name='Location_T', antisym=True), [self.hidden_dim, self.hidden_dim])

        self.gamma2 = tf.get_variable('h2_IN_gamma', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.beta2 = tf.get_variable('h2_IN_beta', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.gamma4 = tf.get_variable('h4_IN_gamma', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.beta4 = tf.get_variable('h4_IN_beta', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
        self.gamma6 = tf.get_variable('h6_IN_gamma', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
        self.beta6 = tf.get_variable('h6_IN_beta', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
        self.gamma8 = tf.get_variable('h8_IN_gamma', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 64]), dtype=tf.float32))
        self.beta8 = tf.get_variable('h8_IN_beta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 64]), dtype=tf.float32))

        def get_R(dtheta):
            return tf.eye(self.hidden_dim) + self.T_loc * dtheta + tf.matmul(self.T_loc, self.T_loc) * (dtheta ** 2) / 2.0

        self.R_list = []
        for i in range(self.num_B_dtheta + 1):
            self.R_list.append(get_R(i * 2 * np.pi / self.num_B_loc))
        self.R_list = tf.stack(self.R_list)

        self.v_ins_reg = self.v_instance / tf.norm(self.v_instance, axis=-1, keep_dims=True)
        self.v_view_reg = self.v_view / tf.norm(self.v_view, axis=-1, keep_dims=True)
        v_loc_reshape = tf.reshape(self.v_location,
                                   (self.location_num + 1, self.location_num + 1, self.num_block, self.block_size))
        v_loc_reg = v_loc_reshape / (tf.norm(v_loc_reshape, axis=-1, keep_dims=True) * self.num_block)
        self.v_loc_reg = tf.reshape(v_loc_reg, (self.location_num + 1, self.location_num + 1, self.hidden_dim))

        # rotation for head system
        v_view0 = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle0, self.in_rest_angle0)
        v_loc0 = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location0, self.in_rest_location0)

        v_view_noise = v_view0 + self.noise_view
        v_loc_noise = v_loc0 + self.noise_loc

        v_ins = tf.gather(self.v_ins_reg, self.in_ins_idx)
        v_total0 = tf.concat([v_ins, v_view_noise, v_loc_noise], axis=-1)
        self.re_img0 = self.decoder(v_total0, self.in_ins_idx, reuse=False)


        self.saver = tf.train.Saver(max_to_keep=20)
        could_load, _ = self.load()
        assert could_load

        # calculate the std of location and angle vectors
        view_vec, loc_vec = self.sess.run([self.v_view_reg, self.v_loc_reg])
        view_vec = np.reshape(view_vec, (-1, self.hidden_dim))
        loc_vec = np.reshape(loc_vec, (-1, self.hidden_dim))
        view_std = np.std(view_vec, axis=0)
        loc_std = np.std(loc_vec, axis=0)

        num_ins = self.data_loader.get_num_ins()
        start_time = time.time()
        avg_loss = {}
        avg_psnr = {}
        b_sz = 200
        count = 0
        for i in range(13):
            for j in range(num_ins):
                avg_loss[(j, i)] = 0.0
                avg_psnr[(j, i)] = 0.0

        for i in range(num_ins):
            imgs, poses = self.data_loader.get_all_test(i)

            locs = poses[:, 0:2]
            angles = poses[:, 2]
            angles /= (180.0 / self.grid_num)
            locs /= self.location_length

            grid_angles, rest_angles = self.quantize_angle(angles, train=False)
            grid_locs, rest_locs = self.quantize_loc(locs, train=False)

            num_img = len(imgs)
            num_batch = math.ceil(float(num_img) / b_sz)

            for k in range(num_batch):
                start = k * b_sz
                end = min(num_img, (k+1) * b_sz)
                cur_img = imgs[start: end]
                cur_grid_angles = grid_angles[start: end]
                cur_rest_angles = rest_angles[start: end]
                cur_grid_locs = grid_locs[start: end]
                cur_rest_locs = rest_locs[start: end]
                cur_idx_in = np.array([i] * (end - start), dtype=np.int)

                for j in range(13):
                    cur_loc_std = 0.5 / 20.0 * j * np.expand_dims(loc_std, axis=0) * np.random.normal(loc=0.0, scale=1.0, size=(end - start, self.hidden_dim))
                    cur_view_std = 0.5 / 20.0 * j * np.expand_dims(view_std, axis=0) * np.random.normal(loc=0.0, scale=1.0, size=(end - start, self.hidden_dim))
                    feed_dict = {self.in_ins_idx: cur_idx_in,
                                 self.in_grid_angle0: cur_grid_angles,
                                 self.in_rest_angle0: cur_rest_angles,
                                 self.in_grid_location0: cur_grid_locs,
                                 self.in_rest_location0: cur_rest_locs,
                                 self.noise_view: cur_view_std,
                                 self.noise_loc: cur_loc_std}

                    re_img = self.sess.run(self.re_img0, feed_dict=feed_dict)
                    recons_psnr = np.sum(calculate_psnr(re_img, cur_img))
                    recons_loss = np.sum(np.square(re_img - cur_img))
                    avg_loss[(i, j)] += recons_loss / num_img
                    avg_psnr[(i, j)] += recons_psnr / num_img
                    if k == 0:
                        save_images(re_img[:64], os.path.join(save_path, '{}_{}_recons.png'.format(i, j)))
                        save_images(cur_img[:64], os.path.join(save_path, '{}_{}_ori.png'.format(i, j)))

                if count % 100 == 0:
                    print(count, time.time() - start_time)
                count += 1

            for j in range(13):
                print("ins {} time {} noise mag {} loss {} psnr {}".format(i, time.time() - start_time, 0.5 / 20 * j, avg_loss[(i, j)], avg_psnr[(i, j)]))

        for j in range(13):
            total_avg_loss = 0.0
            total_avg_psnr = 0.0
            for i in range(num_ins):
                total_avg_psnr += avg_psnr[(i, j)] / float(num_ins)
                total_avg_loss += avg_loss[(i, j)] / float(num_ins)
            print("noise mag {} loss {} psnr {}".format(j * 0.5 / 20, total_avg_loss, total_avg_psnr))

    def train(self):
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(tf.global_variables_initializer())
        num_ins = self.data_loader.get_num_ins()
        could_load, checkpoint_counter = self.load()
        if could_load:
            print(" [*] Load SUCCESS")
            count = checkpoint_counter
        else:
            print(" [!] Load failed...")
            count = 0

        start_time = time.time()
        num_iteration = math.floor(float(num_ins) / self.nins_per_batch)

        bsz = 3000
        avg_rot_view = 0.0
        avg_rot_loc = 0.0
        avg_B_rot = 0.0
        avg_recons = 0.0

        for epoch in range(self.epoch):
            new_idx = np.random.permutation(num_ins)
            for i in range(num_iteration):
                start = i * self.nins_per_batch
                end = min((i+1) * self.nins_per_batch, num_ins)
                cur_ins_idx = new_idx[start: end].astype(np.int)

                cur_img0 = []
                cur_img1 = []
                cur_pos0 = []
                cur_pos1 = []

                # load in image data and their corresponding pose
                for idx in cur_ins_idx:
                    tmp_img, tmp_pos = self.data_loader.get_item(idx, self.npair_per_ins, 'train', use_all=False)
                    tmp_img = np.reshape(tmp_img, (self.npair_per_ins, 2, self.im_sz, self.im_sz, self.channel))
                    tmp_pos = np.reshape(tmp_pos, (self.npair_per_ins, 2, 3))
                    cur_img0.append(tmp_img[:, 0])
                    cur_img1.append(tmp_img[:, 1])
                    cur_pos0.append(tmp_pos[:, 0])
                    cur_pos1.append(tmp_pos[:, 1])
                
                cur_img0 = np.concatenate(cur_img0, axis = 0)
                cur_img1 = np.concatenate(cur_img1, axis = 0)
                cur_pos0 = np.concatenate(cur_pos0, axis = 0)
                cur_pos1 = np.concatenate(cur_pos1, axis = 0)
                cur_angle0 = cur_pos0[:, 2]
                cur_angle1 = cur_pos1[:, 2]
                cur_loc0 = cur_pos0[:, 0:2]
                cur_loc1 = cur_pos1[:, 0:2]

                cur_angle0 /= (180.0 / self.grid_num)
                cur_angle1 /= (180.0 / self.grid_num)
                cur_loc0 /= self.location_length
                cur_loc1 /= self.location_length

                cur_ins_idx = np.reshape(np.tile(np.expand_dims(cur_ins_idx, axis=-1), (1, self.npair_per_ins)), (-1))

                # randomly sample pose data to help training the pose rotation loss
                add_angle0 = np.random.uniform(low=0.0, high=2 * self.grid_num - 1.0, size=(bsz))
                add_angle1 = add_angle0 + np.random.uniform(low=0.5, high=1.0, size=(bsz))
                add_angle1 = np.clip(add_angle1, 0.0, 2 * self.grid_num)
                add_x0 = np.random.uniform(low=1.0, high=self.location_num - 1.0, size=(bsz, 1))
                add_y0 = np.random.uniform(low=1.0, high=self.location_num - 1.0, size=(bsz, 1))
                tmp_direction = np.random.randint(low=0.0, high=self.num_B_loc + 1, size=(bsz, 1)).astype(np.float32) / self.num_B_loc * 2 * np.pi
                tmp_r = np.random.uniform(0.0, 2.2, size=(bsz, 1))
                add_dx = tmp_r * np.cos(tmp_direction)
                add_dy = tmp_r * np.sin(tmp_direction)
                add_x1 = add_x0 + add_dx
                add_x1 = np.clip(add_x1, 0.0, self.location_num)
                add_y1 = add_y0 + add_dy
                add_y1 = np.clip(add_y1, 0.0, self.location_num)
                add_loc0 = np.concatenate([add_x0, add_y0], axis=-1)
                add_loc1 = np.concatenate([add_x1, add_y1], axis=-1)

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
                             self.in_ins_idx: cur_ins_idx,
                             self.in_grid_angle0: cur_grid_angle0,
                             self.in_rest_angle0: cur_rest_angle0,
                             self.in_grid_angle1: cur_grid_angle1,
                             self.in_rest_angle1: cur_rest_angle1,
                             self.add_grid_angle0: add_grid_angle0,
                             self.add_rest_angle0: add_rest_angle0,
                             self.add_grid_angle1: add_grid_angle1,
                             self.add_rest_angle1: add_rest_angle1,
                             self.in_grid_location0: cur_grid_loc0,
                             self.in_grid_location1: cur_grid_loc1,
                             self.in_rest_location0: cur_rest_loc0,
                             self.in_rest_location1: cur_rest_loc1,
                             self.add_grid_location0: add_grid_loc0,
                             self.add_grid_location1: add_grid_loc1,
                             self.add_rest_location0: add_rest_loc0,
                             self.add_rest_location1: add_rest_loc1}

                # update the underlying rotation system
                for _ in range(self.update_step):
                    self.sess.run(self.update_rot, feed_dict=feed_dict)

                # update the decoder parameters
                res = self.sess.run([self.update_dec, self.rot_loss_loc, self.rot_loss_view, self.recons_loss, self.B_rotation_loss, self.re_img0, self.re_img1], feed_dict=feed_dict)
                _, rot_loss_loc, rot_loss_view, recons_loss, B_rot_loss, re_img0, re_img1 = res

                re_img = np.concatenate([np.expand_dims(re_img0, axis = 1), np.expand_dims(re_img1, axis = 1)], axis = 1)
                re_img = np.reshape(re_img, (self.nins_per_batch * self.nimg_per_ins, self.im_sz, self.im_sz, self.channel))
                cur_img = np.concatenate([np.expand_dims(cur_img0, axis = 1), np.expand_dims(cur_img1, axis = 1)], axis = 1)
                cur_img = np.reshape(cur_img, (self.nins_per_batch * self.nimg_per_ins, self.im_sz, self.im_sz, self.channel))
                recons_psnr = np.mean(calculate_psnr(re_img, cur_img))
                avg_rot_loc += rot_loss_loc / self.print_iter
                avg_rot_view += rot_loss_view / self.print_iter
                avg_B_rot += B_rot_loss / self.print_iter
                avg_recons += recons_loss / self.print_iter

                if count % self.print_iter == 0:

                    print('Epoch {} iter {} time {:.3f} rot loc {:.3f} rot view {:.3f} rot B {:4f} recons loss {:.3f} psnr {:.2f}'.\
                          format(epoch, i, time.time() - start_time, avg_rot_loc * self.rot_reg_weight, avg_rot_view * self.rot_reg_weight, \
                                 avg_B_rot * 100, avg_recons, recons_psnr))
                    avg_rot_loc = 0.0
                    avg_rot_view = 0.0
                    avg_B_rot = 0.0
                    avg_recons = 0.0


                if count % (self.print_iter * 4) == 0:
                    save_images(cur_img[:100], os.path.join(self.sample_dir, 'ori{:06d}.png'.format(count)))
                    save_images(re_img[:100], os.path.join(self.sample_dir, 'recons0_img{:06d}.png'.format(count)))

                if count % (self.print_iter * 6) == 0 and count != 0:
                    self.save(count)

                count = count + 1

#########################################  config  #########################################

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--epoch', type=int, default=100000, help='Number of epochs to train')
parser.add_argument('--nins_per_batch', type=int, default=4, help='Number of different instances in one batch')
parser.add_argument('--nimg_per_ins', type=int, default=50, help='Number of image chosen for one instance in one batch, since we use pair of data, should be multiple of 2')
parser.add_argument('--print_iter', type=int, default=500, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # TODO was 0.003
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')
parser.add_argument('--update_step', type=int, default=2, help='Number of inference step in Langevin')
parser.add_argument('--update_step_sz', type=float, default=1e-2, help='Step size for Langevin update')
parser.add_argument('--train', type=bool, default=False, help="Whether we are in train mode or test mode")
# weight of different losses
parser.add_argument('--recons_weight', type=float, default=0.01, help='Reconstruction loss weight')
parser.add_argument('--B_reg_weight', type=float, default=0.8, help='Regulariza the rotation of B')
parser.add_argument('--rot_reg_weight', type=float, default=100.0, help='Regularization weight for whether vectors agree with each other') #90
# structure parameters
parser.add_argument('--num_block', type=int, default=6, help='Number of blocks in the representation')
parser.add_argument('--block_size', type=int, default=16, help='Number of neurons per block')
parser.add_argument('--grid_num', type=int, default=18, help='Number of grid angle')
parser.add_argument('--grid_angle', type=float, default=np.pi/18, help='Size of one angle grid')
parser.add_argument('--num_B_loc', type=int, default=144, help='Number of head rotation')
parser.add_argument('--num_B_dtheta', type=int, default=5, help='Number of dtheta in head rotation regularization')
parser.add_argument('--location_num', type=int, default=40, help='Number of location grids')
parser.add_argument('--location_length', type=float, default=2.0 / 40, help='Length of a grid')
# dataset parameters
parser.add_argument('--im_sz', type=int, default=128, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--data_path', type=str, default='../dataset/gibson_room', help='path for dataset')
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
        model = decoder_model(FLAGS, sess)
        if FLAGS.train:
            model.train()
        else:
            model.test_by_noise('test_noise')

if __name__ == '__main__':
    tf.app.run()



