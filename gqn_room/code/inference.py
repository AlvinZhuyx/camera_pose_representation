from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
from utils import *
import argparse
import math
from data_reader import DataReader
import pickle

##########################################################################################
# build inference model upon learned vector representation
#########################################  data  #########################################

class recons_model(object):
    def __init__(self, FLAGS, sess):
        self.beta1 = FLAGS.beta1
        self.lr = FLAGS.lr
        self.im_sz = FLAGS.im_sz
        self.channel = FLAGS.channel
        self.epoch = FLAGS.epoch
        self.num_block = FLAGS.num_block
        self.block_size = FLAGS.block_size
        self.location_num = FLAGS.location_num
        self.grid_num = FLAGS.grid_num
        self.num_B_loc = FLAGS.num_B_loc
        self.location_length = FLAGS.location_length
        self.grid_angle = FLAGS.grid_angle
        self.hidden_dim = self.num_block * self.block_size
        self.checkpoint_dir_load = FLAGS.checkpoint_dir_load
        self.checkpoint_dir_save = FLAGS.checkpoint_dir_save
        self.sess = sess
        self.nsee_per_ins = FLAGS.nsee_per_ins
        self.npred_per_ins = FLAGS.npred_per_ins
        self.nimg_per_ins = self.nsee_per_ins + self.npred_per_ins
        self.nins_per_batch = FLAGS.nins_per_batch
        self.print_iter = FLAGS.print_iter
        self.path = FLAGS.data_path
        dataset = DataReader(mode='test', dataset='rooms_free_camera_no_object_rotations', context_size=self.nimg_per_ins, root=FLAGS.data_path)
        self.data_loader = dataset.read(batch_size=self.nins_per_batch)

    def inference_model(self, inputs, gamma4, beta4, gamma8, beta8, gamma16, beta16, reuse=False):
        def ins_norm(x, gamma, beta):
            assert len(x.shape) == 4
            C = x.shape[3]
            u = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            sigma = tf.sqrt(tf.reduce_mean(tf.square(x - u), axis=[1, 2], keep_dims=True) + 1e-7)
            gamma = tf.reshape(gamma, (-1, 1, 1, C))
            beta = tf.reshape(beta, (-1, 1, 1, C))
            norm_x = gamma * (x - u) / sigma + beta
            return norm_x

        with tf.variable_scope('infer_model', reuse=reuse):
            # 64 * 64 -- > 32 * 32
            h1 = tf.layers.conv2d(inputs, 64, 4, 2, padding='same', name='conv1')
            h1 = tf.nn.leaky_relu(h1)
            # 32 * 32 --> 16 * 16
            h2 = tf.layers.conv2d(h1, 128, 4, 2, padding='same', name='conv2')
            h2 = ins_norm(h2, gamma16, beta16)
            h2 = tf.nn.leaky_relu(h2)
            # 16 * 16 --> 8 * 8
            h3 = tf.layers.conv2d(h2, 256, 4, 2, padding='same', name='conv3')
            h3 = ins_norm(h3, gamma8, beta8)
            h3 = tf.nn.leaky_relu(h3)
            # 8 * 8 --> 4 * 4
            h4 = tf.layers.conv2d(h3, 256, 4, 2, padding='same', name='conv4')
            h4 = ins_norm(h4, gamma4, beta4)
            h4 = tf.nn.leaky_relu(h4)
            h4 = tf.reshape(h4, (-1, 256 * 4 * 4))
            # 4 * 4 * 256 --> self.hidden dim
            h5_loc = tf.layers.dense(h4, self.hidden_dim, name='loc_head')
            h5_view = tf.layers.dense(h4, self.hidden_dim, name='view_head')

            h6_loc = tf.reshape(h5_loc, (-1, self.num_block, self.block_size))
            h6_loc = h6_loc / (tf.norm(h6_loc, axis=-1, keep_dims=True) * self.num_block)
            h6_loc = tf.reshape(h6_loc, (-1, self.hidden_dim))
            h6_view = h5_view / tf.norm(h5_view, axis=-1, keep_dims=True)
            return h6_loc, h6_view

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
            gamma4 = tf.layers.conv2d(h9, filters=256, kernel_size=3, strides=1, padding='same')
            gamma4 = tf.layers.average_pooling2d(gamma4, pool_size=4, strides=4, padding='same')
            beta4 = tf.layers.conv2d(h9, filters=256, kernel_size=3, strides=1, padding='same')
            beta4 = tf.layers.average_pooling2d(beta4, pool_size=4, strides=4, padding='same')
            gamma8 = tf.layers.conv2d(h7, filters=256, kernel_size=3, strides=1, padding='same')
            gamma8 = tf.layers.average_pooling2d(gamma8, pool_size=8, strides=8, padding='same')
            beta8 = tf.layers.conv2d(h7, filters=256, kernel_size=3, strides=1, padding='same')
            beta8 = tf.layers.average_pooling2d(beta8, pool_size=8, strides=8, padding='same')
            gamma16 = tf.layers.conv2d(h5, filters=128, kernel_size=3, strides=1, padding='same')
            gamma16 = tf.layers.average_pooling2d(gamma16, pool_size=16, strides=16, padding='same')
            beta16 = tf.layers.conv2d(h5, filters=128, kernel_size=3, strides=1, padding='same')
            beta16 = tf.layers.average_pooling2d(beta16, pool_size=16, strides=16, padding='same')
        return gamma4, beta4, gamma8, beta8, gamma16, beta16

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

    def build_model(self):
        self.inputs_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins, self.im_sz, self.im_sz, self.channel])
        self.inputs_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins, self.im_sz, self.im_sz, self.channel])

        # theta1 and theta0 should be converted into grid space
        self.in_grid_angle_see = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.nsee_per_ins], name='in_asee_grid')
        self.in_rest_angle_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins], name='in_asee_rest')
        self.in_grid_location_see = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.nsee_per_ins, 2], name='in_lsee_grid')
        self.in_rest_location_see = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.nsee_per_ins, 2], name='in_lsee_rest')
        self.in_grid_angle_pred = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npred_per_ins], name='in_apred_grid')
        self.in_rest_angle_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins], name='in_apred_rest')
        self.in_grid_location_pred = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npred_per_ins, 2], name='in_lpred_grid')
        self.in_rest_location_pred = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npred_per_ins, 2], name='in_lpred_rest')

        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None], name='in_asee_grid')
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None], name='in_asee_rest')
        self.in_grid_location0 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='in_lsee_grid')
        self.in_rest_location0 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='in_lsee_rest')

        self.v_location = tf.get_variable('Location_v', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]),
            dtype=tf.float32))
        self.v_view = tf.get_variable('Rotation_v', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.B_rot = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block,
                                block_size=self.block_size, name='Rotation_B', antisym=True), [self.hidden_dim, self.hidden_dim])
        self.B_loc = construct_block_diagonal_weights(num_channel=self.num_B_loc, num_block=self.num_block,
                                                      block_size=self.block_size, name='Location_B', antisym=True)
        self.v_view_reg = self.v_view / tf.norm(self.v_view, axis=-1, keep_dims=True)
        v_loc_reshape = tf.reshape(self.v_location, (self.location_num + 1, self.location_num + 1, self.num_block, self.block_size))
        v_loc_reg = v_loc_reshape / (tf.norm(v_loc_reshape, axis=-1, keep_dims=True) * self.num_block)
        self.v_loc_reg = tf.reshape(v_loc_reg, (self.location_num + 1, self.location_num + 1, self.hidden_dim))

        v_view_see = self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle_see, self.in_rest_angle_see)
        v_loc_see = self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location_see, self.in_rest_location_see)
        v_pos_see = tf.concat([v_view_see, v_loc_see], axis=1)
        gamma4, beta4, gamma8, beta8, gamma16, beta16 = self.encoder(self.inputs_see, v_pos_see, reuse=False)

        def inf_ensemble(h_in):
            shape = h_in.shape.as_list()
            assert len(shape) == 4
            assert shape[0] == self.nins_per_batch * self.nsee_per_ins
            h_in = tf.reshape(h_in, (self.nins_per_batch, self.nsee_per_ins, shape[1], shape[2], shape[3]))
            h_in = tf.reduce_sum(h_in, axis=1, keep_dims=True)
            h_in = tf.tile(h_in, (1, self.npred_per_ins, 1, 1, 1))
            h_in = tf.reshape(h_in, (self.nins_per_batch * self.npred_per_ins, shape[1], shape[2], shape[3]))
            return h_in

        gamma4 = inf_ensemble(gamma4)
        beta4 = inf_ensemble(beta4)
        gamma8 = inf_ensemble(gamma8)
        beta8 = inf_ensemble(beta8)
        gamma16 = inf_ensemble(gamma16)
        beta16 = inf_ensemble(beta16)


        # predict vs underlying vector
        self.loc_pred, self.view_pred = self.inference_model(self.inputs_pred, gamma4, beta4, gamma8, beta8, gamma16, beta16, reuse=False)
        self.view_target = tf.stop_gradient(self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle_pred, self.in_rest_angle_pred))
        self.loc_target = tf.stop_gradient(self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location_pred, self.in_rest_location_pred))

        self.view0 = tf.stop_gradient(self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle0, self.in_rest_angle0))
        self.loc0 = tf.stop_gradient(self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location0, self.in_rest_location0))
        self.sax = tf.get_variable('sax', initializer=tf.convert_to_tensor(float(-np.log(20))), dtype=tf.float32, trainable=True)
        self.saq = tf.get_variable('saq', initializer=tf.convert_to_tensor(float(-np.log(5))), dtype=tf.float32, trainable=True)
        self.loc_loss = tf.exp(-self.sax) * tf.reduce_sum(tf.reduce_mean(tf.square(self.loc_target - self.loc_pred), axis=0)) + self.sax
        self.view_loss = tf.exp(-self.saq) * tf.reduce_sum(tf.reduce_mean(tf.square(self.view_target - self.view_pred), axis=0)) + self.saq

        update_var = []
        for var in tf.trainable_variables():
            if "infer_model" in var.name or 'encoder' in var.name or 'sa' in var.name:
                print(var.name, var.shape)
                update_var.append(var)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9).minimize(self.loc_loss + self.view_loss, var_list=update_var)

    def save(self, step):
        model_name = 'infer.model'
        checkpoint_dir = self.checkpoint_dir_save
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def quantize_angle(self, angle):
        assert np.max(angle) <= 2 * self.grid_num and np.max(angle) >= 0
        grid_angle = np.floor(angle)
        grid_angle = np.clip(grid_angle, 0, 2 * self.grid_num)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def quantize_loc(self, loc):
        assert np.max(loc) < self.location_num + 1 and np.min(loc) >= 0
        grid_loc = np.floor(loc)
        grid_loc = np.clip(grid_loc, 0, self.location_num)
        rest_loc = loc - grid_loc.astype(np.float32)
        return grid_loc.astype(np.int), rest_loc.astype(np.float32)

    def load(self, checkpoint_dir, saver):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        # the hyperparameters depends on the dataset we split out
        from room_dataset_tf import RoomDataset, transform_img, sample_batch
        from torch.utils.data import DataLoader
        self.nins_per_batch = 64
        self.nsee_per_ins = 6
        self.npred_per_ins = 1
        self.nimg_per_ins = 7
        self.build_model()

        load_var_list = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name or 'Location' in var.name:
                print('Load variable from generator: ', var.name)
                load_var_list.append(var)
        self.saver_load = tf.train.Saver(max_to_keep=20, var_list=load_var_list)
        self.saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(tf.global_variables_initializer())
        print("Load rotation system from generator checkpoint")
        could_load, checkpoint_counter = self.load(self.checkpoint_dir_load, self.saver_load)
        assert could_load
        print("Try to load pretrained inference model")
        could_load, checkpoint_counter = self.load(self.checkpoint_dir_save, self.saver)
        assert could_load

        # decoding system
        x, y = np.meshgrid(np.arange(0, self.location_num + 0.25, 0.25), np.arange(0, self.location_num + 0.25, 0.25))
        theta = np.arange(0, 360, 1)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        loc_in = np.concatenate([x, y], axis=-1)
        theta_in = theta / (180 / self.grid_num)
        loc_grid_in = np.floor(loc_in)
        loc_rest_in = loc_in - loc_grid_in
        theta_grid_in = np.floor(theta_in)
        theta_rest_in = theta_in - theta_grid_in
        feed_dict = {
            self.in_grid_location0: loc_grid_in,
            self.in_rest_location0: loc_rest_in,
            self.in_grid_angle0: theta_grid_in,
            self.in_rest_angle0: theta_rest_in
        }
        v_loc_list, v_view_list = self.sess.run([self.loc0, self.view0], feed_dict=feed_dict)

        # begin testing
        test_dataset = RoomDataset(root_dir="../dataset/gqn_room/torch", transform=transform_img)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_iter = iter(test_loader)
        start_time = time.time()
        total_data = 0.0
        avg_x_diff = 0.0
        avg_y_diff = 0.0
        avg_theta_diff = 0.0
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

            feed_dict = {
                self.inputs_pred: cur_img_pred,
                self.inputs_see: cur_img_see,
                self.in_grid_angle_see: cur_grid_angle_see,
                self.in_rest_angle_see: cur_rest_angle_see,
                self.in_grid_angle_pred: cur_grid_angle_pred,
                self.in_rest_angle_pred: cur_rest_angle_pred,
                self.in_grid_location_see: cur_grid_loc_see,
                self.in_rest_location_see: cur_rest_loc_see,
                self.in_grid_location_pred: cur_grid_loc_pred,
                self.in_rest_location_pred: cur_rest_loc_pred,
            }
            loc_pred_vec, view_pred_vec = self.sess.run([self.loc_pred, self.view_pred], feed_dict=feed_dict)
            loc_pred_vec = loc_pred_vec[: cur_test_num]
            view_pred_vec = view_pred_vec[: cur_test_num]

            loc_vec_diff = np.sum(np.square(np.expand_dims(loc_pred_vec, axis=1) - np.expand_dims(v_loc_list, axis=0)), axis=-1)
            loc_idx = np.argmin(loc_vec_diff, axis=-1)
            x_infer = np.squeeze(x[loc_idx])
            y_infer = np.squeeze(y[loc_idx])
            view_vec_diff = np.sum(np.square(np.expand_dims(view_pred_vec, axis=1) - np.expand_dims(v_view_list, axis=0)), axis=-1)
            view_idx = np.argmin(view_vec_diff, axis=-1)
            theta_infer = theta[view_idx]
            x_diff = np.sum(np.abs(x_infer - cur_loc_pred[: cur_test_num, 0])) * self.location_length
            y_diff = np.sum(np.abs(y_infer - cur_loc_pred[: cur_test_num, 1])) * self.location_length
            theta_diff = theta_infer - cur_view_pred[: cur_test_num] * 180 / self.grid_num
            theta_diff[theta_diff > 180] -= 360
            theta_diff[theta_diff < -180] += 360
            theta_diff = np.sum(np.abs(theta_diff))

            avg_x_diff += x_diff
            avg_y_diff += y_diff
            avg_theta_diff += theta_diff

            if count % 100 == 0:
                print("Finish {}/18900, time {:.3f}, current result x_diff {:.3f}, y_diff {:.3f}, theta_diff {:.3f}" .format(count, time.time() - start_time, avg_x_diff/total_data, avg_y_diff/total_data, avg_theta_diff/total_data))

            count += 1

        avg_x_diff /= total_data
        avg_y_diff /= total_data
        avg_theta_diff /= total_data
        print("final result x diff {:.4f} y diff {:.4f} theta diff {:.4f}".format(avg_x_diff, avg_y_diff, avg_theta_diff))


    def train(self):
        self.build_model()
        load_var_list = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name or 'Location' in var.name:
                print('Load variable from generator: ', var.name)
                load_var_list.append(var)
        self.saver_load = tf.train.Saver(max_to_keep=5, var_list=load_var_list)
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())
        print("Load rotation system from generator checkpoint")
        could_load, checkpoint_counter = self.load(self.checkpoint_dir_load, self.saver_load)
        assert could_load
        print("Try to load pretrained inference model")
        could_load, checkpoint_counter = self.load(self.checkpoint_dir_save, self.saver)
        if could_load:
            print(" [*] Load SUCCESS")
            count = checkpoint_counter
        else:
            print(" [!] Load failed...")
            count = 0

        self.sess_data = tf.train.SingularMonitoredSession()
        start_time = time.time()
        avg_loc_loss = 0.0
        avg_view_loss = 0.0

        # decoding system
        x, y = np.meshgrid(np.arange(0, self.location_num + 0.5, 0.5), np.arange(0, self.location_num + 0.5, 0.5))
        theta = np.arange(0, 360, 1)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        loc_in = np.concatenate([x, y], axis=-1)
        theta_in = theta / (180 / self.grid_num)
        loc_grid_in = np.floor(loc_in)
        loc_rest_in = loc_in - loc_grid_in
        theta_grid_in = np.floor(theta_in)
        theta_rest_in = theta_in - theta_grid_in
        feed_dict = {
            self.in_grid_location0: loc_grid_in,
            self.in_rest_location0: loc_rest_in,
            self.in_grid_angle0: theta_grid_in,
            self.in_rest_angle0: theta_rest_in
        }
        v_loc_list, v_view_list = self.sess.run([self.loc0, self.view0], feed_dict=feed_dict)



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

            cur_grid_angle_see, cur_rest_angle_see = self.quantize_angle(angle_see)
            cur_grid_angle_pred, cur_rest_angle_pred = self.quantize_angle(angle_pred)
            cur_grid_loc_see, cur_rest_loc_see = self.quantize_loc(loc_see)
            cur_grid_loc_pred, cur_rest_loc_pred = self.quantize_loc(loc_pred)



            feed_dict = {
                self.inputs_pred: img_pred,
                self.inputs_see: img_see,
                self.in_grid_angle_see: cur_grid_angle_see,
                self.in_rest_angle_see: cur_rest_angle_see,
                self.in_grid_angle_pred: cur_grid_angle_pred,
                self.in_rest_angle_pred: cur_rest_angle_pred,
                self.in_grid_location_see: cur_grid_loc_see,
                self.in_rest_location_see: cur_rest_loc_see,
                self.in_grid_location_pred: cur_grid_loc_pred,
                self.in_rest_location_pred: cur_rest_loc_pred,
            }
            _, loc_loss, view_loss, loc_pred_vec, view_pred_vec = self.sess.run([self.optim, self.loc_loss, self.view_loss, self.loc_pred, self.view_pred], feed_dict=feed_dict)

            avg_loc_loss += loc_loss / self.print_iter
            avg_view_loss += view_loss / self.print_iter


            loc_vec_diff = np.sum(np.square(np.expand_dims(loc_pred_vec, axis=1) - np.expand_dims(v_loc_list, axis=0)), axis=-1)
            loc_idx = np.argmin(loc_vec_diff, axis=-1)
            x_infer = np.squeeze(x[loc_idx])
            y_infer = np.squeeze(y[loc_idx])
            view_vec_diff = np.sum(np.square(np.expand_dims(view_pred_vec, axis=1) - np.expand_dims(v_view_list, axis=0)), axis=-1)
            view_idx = np.argmin(view_vec_diff, axis=-1)
            theta_infer = theta[view_idx]
            x_diff = np.mean(np.abs(x_infer - loc_pred[:, 0])) * self.location_length
            y_diff = np.mean(np.abs(y_infer - loc_pred[:, 1])) * self.location_length
            theta_diff = theta_infer - angle_pred * 180 / self.grid_num
            theta_diff[theta_diff > 180] -= 360
            theta_diff[theta_diff < -180] += 360
            theta_diff = np.mean(np.abs(theta_diff))
            if count % self.print_iter == 0:
                print("Epoch{}, time {:.3f}, avg loc loss {:.3f}, avg view loss {:.3f}, x_diff {:.3f}, y_diff {:.3f}, theta_diff {:.3f}" \
                      .format(epoch, time.time() - start_time, avg_loc_loss, avg_view_loss, x_diff, y_diff, theta_diff))
                avg_loc_loss = 0.0
                avg_view_loss = 0.0

            # do test
            if count % (self.print_iter * 5) == 0:
                # test on current batch's query image
                # tile this image to fit the placholder
                query_frame_tile = np.reshape(np.tile(np.expand_dims(query_frame, axis=1), (1, self.npred_per_ins, 1, 1, 1)), (-1, self.im_sz, self.im_sz, self.channel))
                query_camera_tile = np.reshape(np.tile(np.expand_dims(query_camera, axis=1), (1, self.npred_per_ins, 1)), (-1, 5))
                query_loc = query_camera_tile[:, :2]
                query_view = query_camera_tile[:, 3]
                query_loc = (query_loc + 1.0) / self.location_length
                query_view = (query_view + 2 * np.pi) / self.grid_angle
                cur_grid_angle_pred, cur_rest_angle_pred = self.quantize_angle(query_view)
                cur_grid_loc_pred, cur_rest_loc_pred = self.quantize_loc(query_loc)

                feed_dict = {
                    self.inputs_pred: query_frame_tile,
                    self.inputs_see: img_see,
                    self.in_grid_angle_see: cur_grid_angle_see,
                    self.in_rest_angle_see: cur_rest_angle_see,
                    self.in_grid_angle_pred: cur_grid_angle_pred,
                    self.in_rest_angle_pred: cur_rest_angle_pred,
                    self.in_grid_location_see: cur_grid_loc_see,
                    self.in_rest_location_see: cur_rest_loc_see,
                    self.in_grid_location_pred: cur_grid_loc_pred,
                    self.in_rest_location_pred: cur_rest_loc_pred,
                }
                loc_loss, view_loss, loc_pred_vec, view_pred_vec = self.sess.run([self.loc_loss, self.view_loss, self.loc_pred, self.view_pred], feed_dict=feed_dict)
                loc_vec_diff = np.sum(np.square(np.expand_dims(loc_pred_vec, axis=1) - np.expand_dims(v_loc_list, axis=0)), axis=-1)
                loc_idx = np.argmin(loc_vec_diff, axis=-1)
                x_infer = np.squeeze(x[loc_idx])
                y_infer = np.squeeze(y[loc_idx])
                view_vec_diff = np.sum(np.square(np.expand_dims(view_pred_vec, axis=1) - np.expand_dims(v_view_list, axis=0)), axis=-1)
                view_idx = np.argmin(view_vec_diff, axis=-1)
                theta_infer = theta[view_idx]
                x_diff = np.mean(np.abs(x_infer - query_loc[:, 0])) * self.location_length
                y_diff = np.mean(np.abs(y_infer - query_loc[:, 1])) * self.location_length

                theta_diff = theta_infer - query_view * 180 / self.grid_num
                theta_diff[theta_diff > 180] -= 360
                theta_diff[theta_diff < -180] += 360
                theta_diff = np.mean(np.abs(theta_diff))

                print("Do test: avg loc loss {:.3f}, avg view loss {:.3f}, x_diff {:.3f}, y_diff {:.3f}, theta_diff {:.3f}".format(loc_loss, view_loss, x_diff, y_diff, theta_diff))

            if count % (self.print_iter * 30) == 0 or count == 99900:
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
parser.add_argument('--nins_per_batch', type=int, default=30, help='Number of different instances in one batch')
parser.add_argument('--nsee_per_ins', type=int, default=6, help='Number of image chosen for one instance in one batch')
parser.add_argument('--npred_per_ins', type=int, default=3, help='Number of image chosen for one instance in one batch')
parser.add_argument('--print_iter', type=int, default=100, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # TODO was 0.003
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')
parser.add_argument('--train', type=bool, default=False, help='Whether to train or to test the model')

# structure parameters
parser.add_argument('--num_block', type=int, default=6, help='size of hidden dimension')
parser.add_argument('--block_size', type=int, default=16, help='size of hidden dimension')
parser.add_argument('--grid_num', type=int, default=18, help='Number of grid angle')
parser.add_argument('--grid_angle', type=float, default=np.pi/18, help='Size of one angle grid')
parser.add_argument('--num_B_loc', type=int, default=144, help='Number of head rotation')
parser.add_argument('--location_num', type=int, default=20, help='Number of location grids')
parser.add_argument('--location_length', type=float, default=2.0 / 20, help='Length of a grid')
parser.add_argument('--im_sz', type=int, default=64, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--data_path', type=str, default='../dataset/gqn_room', help='path for dataset')
parser.add_argument('--checkpoint_dir_save', type=str, default='./checkpoint_infer', help='path for saving checkpoint')
parser.add_argument('--checkpoint_dir_load', type=str, default='./checkpoint', help='path for saving checkpoint')

FLAGS = parser.parse_args()
def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.set_random_seed(2345)
    np.random.seed(1608)

    if not os.path.exists(FLAGS.checkpoint_dir_save):
        os.makedirs(FLAGS.checkpoint_dir_save)
    with tf.Session() as sess:
        model = recons_model(FLAGS, sess)
        if FLAGS.train:
            model.train()
        else:
            model.test()
if __name__ == '__main__':
    tf.app.run()
