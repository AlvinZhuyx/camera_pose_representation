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
import pickle

##########################################################################################
# build inference model upon learned vector representation
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
        # randomly load num_pair images from the idx instance
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
        self.nimg_per_ins = FLAGS.nimg_per_ins
        self.nins_per_batch = FLAGS.nins_per_batch
        self.print_iter = FLAGS.print_iter
        self.path = FLAGS.data_path
        self.data_loader = dataloader(self.path, im_sz=self.im_sz)

    def inference_model(self, inputs, ins_idx, reuse=False):
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
            # 128 * 128 --> 64 * 64
            if reuse == False:
                num_ins = self.data_loader.get_num_ins()
                self.gamma1 = tf.get_variable('h1_IN_gamma', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 64]), dtype=tf.float32))
                self.beta1 = tf.get_variable('h1_IN_beta', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 64]), dtype=tf.float32))
                self.gamma2 = tf.get_variable('h2_IN_gamma', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
                self.beta2 = tf.get_variable('h2_IN_beta', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
                self.gamma3 = tf.get_variable('h3_IN_gamma', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
                self.beta3 = tf.get_variable('h3_IN_beta', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
                self.gamma4 = tf.get_variable('h4_IN_gamma', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
                self.beta4 = tf.get_variable('h4_IN_beta', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
                self.gamma5 = tf.get_variable('h5_IN_gamma', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))
                self.beta5 = tf.get_variable('h5_IN_beta', initializer=tf.convert_to_tensor(
                    np.random.normal(scale=0.001, size=[num_ins, 256]), dtype=tf.float32))

            h1 = tf.layers.conv2d(inputs, 64, 4, 2, padding='same', name='conv1')
            gamma1 = tf.gather(self.gamma1, ins_idx)
            beta1 = tf.gather(self.beta1, ins_idx)
            h1 = ins_norm(h1, gamma1, beta1)
            h1 = tf.nn.leaky_relu(h1)
            # 64 * 64 -- > 32 * 32
            h2 = tf.layers.conv2d(h1, 128, 4, 2, padding='same', name='conv2')
            gamma2 = tf.gather(self.gamma2, ins_idx)
            beta2 = tf.gather(self.beta2, ins_idx)
            h2 = ins_norm(h2, gamma2, beta2)
            h2 = tf.nn.leaky_relu(h2)
            # 32 * 32 --> 16 * 16
            h3 = tf.layers.conv2d(h2, 256, 4, 2, padding='same', name='conv3')
            gamma3 = tf.gather(self.gamma3, ins_idx)
            beta3 = tf.gather(self.beta3, ins_idx)
            h3 = ins_norm(h3, gamma3, beta3)
            h3 = tf.nn.leaky_relu(h3)
            # 16 * 16 --> 8 * 8
            h4 = tf.layers.conv2d(h3, 256, 4, 2, padding='same', name='conv4')
            gamma4 = tf.gather(self.gamma4, ins_idx)
            beta4 = tf.gather(self.beta4, ins_idx)
            h4 = ins_norm(h4, gamma4, beta4)
            h4 = tf.nn.leaky_relu(h4)
            # 8 * 8 --> 4 * 4
            h5 = tf.layers.conv2d(h4, 256, 4, 2, padding='same', name='conv5')
            gamma5 = tf.gather(self.gamma5, ins_idx)
            beta5 = tf.gather(self.beta5, ins_idx)
            h5 = ins_norm(h5, gamma5, beta5)
            h5 = tf.nn.leaky_relu(h5)
            h5 = tf.reshape(h5, (-1, 256 * 4 * 4))

            # 4 * 4 * 512 --> 256 * 2 * 2
            h6_loc = tf.layers.dense(h5, self.hidden_dim, name='loc_head')
            h6_view = tf.layers.dense(h5, self.hidden_dim, name='view_head')

            # for location vector in Polar Coordinate System, we normalize the vector by each block
            # for orientation vector, we normalize it together
            h6_loc = tf.reshape(h6_loc, (-1, self.num_block, self.block_size))
            h6_loc = h6_loc / (tf.norm(h6_loc, axis=-1, keep_dims=True) * self.num_block)
            h6_loc = tf.reshape(h6_loc, (-1, self.hidden_dim))
            h6_view = h6_view / tf.norm(h6_view, axis=-1, keep_dims=True)
            return h6_loc, h6_view

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
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.im_sz, self.im_sz, self.channel])
        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[None])

        # the pose representation system
        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None], name='in_a0_grid')
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None], name='in_a0_rest')
        self.in_grid_location0 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='in_l0_grid')
        self.in_rest_location0 = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='in_l0_rest')
        self.v_location = tf.get_variable('Location_v', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[self.location_num + 1, self.location_num + 1, self.hidden_dim]),
            dtype=tf.float32))
        self.v_view = tf.get_variable('Rotation_v', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.B_rot = tf.reshape(construct_block_diagonal_weights(num_channel=1, num_block=self.num_block,
                                block_size=self.block_size, name='Rotation_B', antisym=True), [self.hidden_dim, self.hidden_dim])
        self.B_loc = construct_block_diagonal_weights(num_channel=self.num_B_loc, num_block=self.num_block,
                                                      block_size=self.block_size, name='Location_B', antisym=True)

        # for location vector in Polar Coordinate System, we normalize the vector by each block
        # for orientation vector, we normalize it together
        self.v_view_reg = self.v_view / tf.norm(self.v_view, axis=-1, keep_dims=True)
        v_loc_reshape = tf.reshape(self.v_location, (self.location_num + 1, self.location_num + 1, self.num_block, self.block_size))
        v_loc_reg = v_loc_reshape / (tf.norm(v_loc_reshape, axis=-1, keep_dims=True) * self.num_block)
        self.v_loc_reg = tf.reshape(v_loc_reg, (self.location_num + 1, self.location_num + 1, self.hidden_dim))

        # the inference model and loss function definition
        self.loc_pred, self.view_pred = self.inference_model(self.inputs, self.in_ins_idx, reuse=False)
        self.view_target = tf.stop_gradient(self.get_grid_code_rot(self.v_view_reg, self.in_grid_angle0, self.in_rest_angle0))
        self.loc_target = tf.stop_gradient(self.get_grid_code_loc(self.v_loc_reg, self.in_grid_location0, self.in_rest_location0))


        self.loc_loss = 20 * tf.reduce_sum(tf.reduce_mean(tf.square(self.loc_target - self.loc_pred), axis=0))
        self.view_loss = 10 * tf.reduce_sum(tf.reduce_mean(tf.square(self.view_target - self.view_pred), axis=0))

        update_var = []
        for var in tf.trainable_variables():
            if "infer_model" in var.name:
                print(var.name, var.shape)
                update_var.append(var)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9).minimize(self.loc_loss + self.view_loss, var_list=update_var)


    def save(self, step):
        model_name = 'infer.model'
        checkpoint_dir = self.checkpoint_dir_save
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, dir, saver):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = dir
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
        num_ins = self.data_loader.get_num_ins()

        # decoding system
        x, y = np.meshgrid(np.arange(0, self.location_num + 0.5, 0.5), np.arange(0, self.location_num + 0.5, 0.5))
        theta = np.arange(0, 360, 0.25)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        loc_in = np.concatenate([x, y], axis=-1)
        theta_in = theta / (180 / self.grid_num)
        loc_grid_in = np.around(loc_in)
        loc_rest_in = loc_in - loc_grid_in
        theta_grid_in = np.around(theta_in)
        theta_rest_in = theta_in - theta_grid_in
        feed_dict = {
            self.in_grid_location0: loc_grid_in,
            self.in_rest_location0: loc_rest_in,
            self.in_grid_angle0: theta_grid_in,
            self.in_rest_angle0: theta_rest_in
        }
        v_loc_list, v_view_list = self.sess.run([self.loc_target, self.view_target], feed_dict=feed_dict)

        total_data = 0.0
        avg_x_diff = 0.0
        avg_y_diff = 0.0
        avg_theta_diff = 0.0
        start_time = time.time()
        count = 0
        b_sz = 150
        for i in range(num_ins):
            imgs, poses = self.data_loader.get_all_test(i)
            locs = poses[:, 0:2]
            angles = poses[:, 2] / (180.0 / self.grid_num)
            locs /= self.location_length

            grid_locs = np.around(locs)
            rest_locs = locs - grid_locs
            grid_angles = np.around(angles)
            rest_angles = angles - grid_angles

            num_img = len(imgs)
            num_batch = math.ceil(float(num_img) / b_sz)

            for k in range(num_batch):
                start = k * b_sz
                end = min(num_img, (k + 1) * b_sz)
                cur_img = imgs[start: end]
                cur_grid_angles = grid_angles[start: end]
                cur_rest_angles = rest_angles[start: end]
                cur_grid_locs = grid_locs[start: end]
                cur_rest_locs = rest_locs[start: end]
                cur_idx_in = np.array([i] * (end - start), dtype=np.int)
                feed_dict = {
                    self.inputs: cur_img,
                    self.in_ins_idx: cur_idx_in,
                    self.in_grid_angle0: cur_grid_angles,
                    self.in_rest_angle0: cur_rest_angles,
                    self.in_grid_location0: cur_grid_locs,
                    self.in_rest_location0: cur_rest_locs
                }
                loc_pred, view_pred = self.sess.run([self.loc_pred, self.view_pred], feed_dict=feed_dict)
                loc_vec_diff = np.sum(np.square(np.expand_dims(loc_pred, axis=1) - np.expand_dims(v_loc_list, axis=0)), axis=-1)
                loc_idx = np.argmin(loc_vec_diff, axis=-1)
                x_infer = np.squeeze(x[loc_idx])
                y_infer = np.squeeze(y[loc_idx])
                view_vec_diff = np.sum(np.square(np.expand_dims(view_pred, axis=1) - np.expand_dims(v_view_list, axis=0)), axis=-1)
                view_idx = np.argmin(view_vec_diff, axis=-1)
                theta_infer = theta[view_idx]
                x_diff = np.sum(np.abs(x_infer - locs[start: end, 0]))
                y_diff = np.sum(np.abs(y_infer - locs[start: end, 1]))
                theta_diff = theta_infer - poses[start: end, 2]

                theta_diff[theta_diff > 180] -= 360
                theta_diff[theta_diff < -180] += 360
                theta_diff = np.sum(np.abs(theta_diff))
                avg_x_diff += x_diff
                avg_y_diff += y_diff
                avg_theta_diff += theta_diff
                total_data += end - start

                if count % 100 == 0:
                    print("Epoch{}, ins{}, time {:.3f}, x_diff {:.3f}, y_diff {:.3f}, theta_diff {:.3f}".format(count, i, time.time() - start_time,\
                         avg_x_diff / total_data * self.location_length, avg_y_diff / total_data * self.location_length, avg_theta_diff / total_data))
                count += 1
        avg_x_diff /= total_data
        avg_y_diff /= total_data
        avg_theta_diff /= total_data
        print("final result x diff {} y diff {} theta diff {}".format(avg_x_diff * self.location_length, avg_y_diff * self.location_length, avg_theta_diff))


    def train(self):
        self.build_model()
        load_var_list = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name or 'Location' in var.name:
                print('Load variable from generator: ', var.name)
                load_var_list.append(var)
        # load in pretrained pose representation system
        self.saver_load = tf.train.Saver(max_to_keep=1, var_list=load_var_list)
        self.saver = tf.train.Saver(max_to_keep=20)
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
        num_ins = self.data_loader.get_num_ins()
        n_batch = math.ceil(float(num_ins) / self.nins_per_batch)
        start_time = time.time()
        avg_loc_loss = 0.0
        avg_view_loss = 0.0

        # build decoding system (i.e. get the pose vector at each location for decoding)
        x, y = np.meshgrid(np.arange(0, self.location_num + 0.5, 0.5), np.arange(0, self.location_num + 0.5, 0.5))
        theta = np.arange(0, 360, 1)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        loc_in = np.concatenate([x, y], axis=-1)
        theta_in = theta / (180 / self.grid_num)
        loc_grid_in = np.around(loc_in)
        loc_rest_in = loc_in - loc_grid_in
        theta_grid_in = np.around(theta_in)
        theta_rest_in = theta_in - theta_grid_in
        feed_dict = {
            self.in_grid_location0: loc_grid_in,
            self.in_rest_location0: loc_rest_in,
            self.in_grid_angle0: theta_grid_in,
            self.in_rest_angle0: theta_rest_in
        }
        v_loc_list, v_view_list = self.sess.run([self.loc_target, self.view_target], feed_dict=feed_dict)


        for epoch in range(self.epoch):
            new_idx = np.random.permutation(num_ins)
            for i in range(n_batch):
                start = i * self.nins_per_batch
                end = min((i + 1) * self.nins_per_batch, num_ins)
                cur_ins_idx = new_idx[start: end].astype(np.int)

                cur_img = []
                cur_pose = []
                for idx in cur_ins_idx:
                    tmp_img, tmp_pos = self.data_loader.get_item(idx, self.nimg_per_ins // 2, 'train', use_all=False)
                    cur_img.append(tmp_img)
                    cur_pose.append(tmp_pos)

                cur_img = np.concatenate(cur_img, axis=0)
                cur_pose = np.concatenate(cur_pose, axis=0)
                cur_loc = cur_pose[:, :2] / self.location_length
                cur_theta = cur_pose[:, 2] / (180 / self.grid_num)
                cur_grid_loc = np.around(cur_loc)
                cur_rest_loc = cur_loc - cur_grid_loc
                cur_grid_theta = np.around(cur_theta)
                cur_rest_theta = cur_theta - cur_grid_theta
                in_idx = np.reshape(np.tile(np.expand_dims(cur_ins_idx, axis=-1), (1, self.nimg_per_ins)), (-1))
                feed_dict = {
                    self.inputs: cur_img,
                    self.in_ins_idx: in_idx,
                    self.in_grid_angle0: cur_grid_theta,
                    self.in_rest_angle0: cur_rest_theta,
                    self.in_grid_location0: cur_grid_loc,
                    self.in_rest_location0: cur_rest_loc
                }
                _, loc_loss, view_loss, loc_pred, view_pred = self.sess.run([self.optim, self.loc_loss, self.view_loss, self.loc_pred, self.view_pred], feed_dict=feed_dict)

                avg_loc_loss += loc_loss / self.print_iter
                avg_view_loss += view_loss / self.print_iter

                # decoding the predicted pose from pose vector
                loc_vec_diff = np.sum(np.square(np.expand_dims(loc_pred, axis=1) - np.expand_dims(v_loc_list, axis=0)), axis=-1)
                loc_idx = np.argmin(loc_vec_diff, axis=-1)
                x_infer = np.squeeze(x[loc_idx])
                y_infer = np.squeeze(y[loc_idx])
                view_vec_diff = np.sum(np.square(np.expand_dims(view_pred, axis=1) - np.expand_dims(v_view_list, axis=0)), axis=-1)
                view_idx = np.argmin(view_vec_diff, axis=-1)
                theta_infer = theta[view_idx]
                x_diff = np.mean(np.abs(x_infer - cur_loc[:, 0]))
                y_diff = np.mean(np.abs(y_infer - cur_loc[:, 1]))
                theta_diff = theta_infer - cur_pose[:, 2]

                # deal with the repetition of 360 degree when calculating the error, e.g. if the target of the angle is
                # 1 degree and prediction is 359 degree (which is the same with -1 degree), then the error should be 2 degree
                # instead of 358 degree. i.e. we need to calculate the shortest distance between prediction and target on
                # the circle of 360 degree. If more than one orientaiton angle is included here, we need to calculate their distance
                # on a sphere surface

                theta_diff[theta_diff > 180] -= 360
                theta_diff[theta_diff < -180] += 360
                theta_diff = np.mean(np.abs(theta_diff))
                if count % self.print_iter == 0:
                    print("Epoch{}, iter {}, time {:.3f}, avg loc loss {:.3f}, avg view loss {:.3f}, x_diff {:.3f}, y_diff {:.3f}, theta_diff {:.3f}" \
                          .format(epoch, i, time.time() - start_time, avg_loc_loss, avg_view_loss, x_diff * self.location_length, y_diff * self.location_length, theta_diff))
                    avg_loc_loss = 0.0
                    avg_view_loss = 0.0

                # do test on part of the whole test dataset to monitor the training
                if count % (self.print_iter * 5) == 0:
                    x_diff_list = []
                    y_diff_list = []
                    theta_diff_list = []
                    loc_loss_list = []
                    view_loss_list = []
                    for idx in range(num_ins):
                        valid_img, valid_pose = self.data_loader.get_item(idx, 200, 'test', use_all=False)
                        valid_loc = valid_pose[:, :2] / self.location_length
                        valid_theta = valid_pose[:, 2] / (180 / self.grid_num)
                        valid_grid_loc = np.around(valid_loc)
                        valid_rest_loc = valid_loc - valid_grid_loc
                        valid_grid_theta = np.around(valid_theta)
                        valid_rest_theta = valid_theta - valid_grid_theta
                        valid_idx = np.array([idx] * len(valid_img))
                        feed_dict = {
                            self.inputs: valid_img,
                            self.in_ins_idx: valid_idx,
                            self.in_grid_angle0: valid_grid_theta,
                            self.in_rest_angle0: valid_rest_theta,
                            self.in_grid_location0: valid_grid_loc,
                            self.in_rest_location0: valid_rest_loc
                        }
                        loc_loss, view_loss, loc_pred, view_pred = self.sess.run([self.loc_loss, self.view_loss, self.loc_pred, self.view_pred], feed_dict=feed_dict)
                        loc_vec_diff = np.sum(np.square(np.expand_dims(loc_pred, axis=1) - np.expand_dims(v_loc_list, axis=0)), axis=-1)
                        loc_idx = np.argmin(loc_vec_diff, axis=-1)
                        x_infer = np.squeeze(x[loc_idx])
                        y_infer = np.squeeze(y[loc_idx])
                        view_vec_diff = np.sum(np.square(np.expand_dims(view_pred, axis=1) - np.expand_dims(v_view_list, axis=0)), axis=-1)
                        view_idx = np.argmin(view_vec_diff, axis=-1)
                        theta_infer = theta[view_idx]
                        x_diff = np.mean(np.abs(x_infer - valid_loc[:, 0]))
                        y_diff = np.mean(np.abs(y_infer - valid_loc[:, 1]))
                        theta_diff = theta_infer - valid_pose[:, 2]
                        theta_diff[theta_diff > 180] -= 360
                        theta_diff[theta_diff < -180] += 360
                        theta_diff = np.mean(np.abs(theta_diff))
                        x_diff_list.append(x_diff)
                        y_diff_list.append(y_diff)
                        theta_diff_list.append(theta_diff)
                        loc_loss_list.append(loc_loss)
                        view_loss_list.append(view_loss)

                    print("Do test: avg loc loss {:.3f}, avg view loss {:.3f}, x_diff {:.3f}, y_diff {:.3f}, theta_diff {:.3f}"\
                        .format(np.mean(loc_loss_list), np.mean(view_loss_list), np.mean(x_diff_list) * self.location_length, np.mean(y_diff_list) * self.location_length, np.mean(theta_diff_list)))

                if count % (self.print_iter * 9) == 0:
                    self.save(count)
                count = count + 1

#########################################  config  #########################################

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--epoch', type=int, default=5000, help='Number of epochs to train')
parser.add_argument('--nins_per_batch', type=int, default=4, help='Number of different instances in one batch')
parser.add_argument('--nimg_per_ins', type=int, default=50, help='Number of image chosen for one instance in one batch')
parser.add_argument('--print_iter', type=int, default=50, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')  # TODO was 0.003
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')
parser.add_argument('--train', type=bool, default=False, help="Whether we are in train mode or test mode")
# structure parameters
parser.add_argument('--num_block', type=int, default=6, help='size of hidden dimension')
parser.add_argument('--block_size', type=int, default=16, help='size of hidden dimension')
parser.add_argument('--grid_num', type=int, default=18, help='Number of grid angle in 180 degree')
parser.add_argument('--grid_angle', type=float, default=np.pi/18, help='Size of one angle grid')
parser.add_argument('--num_B_loc', type=int, default=144, help='Number of head rotation')
parser.add_argument('--location_num', type=int, default=40, help='Number of location grids')
parser.add_argument('--location_length', type=float, default=2.0 / 40, help='Length of a location grid')
parser.add_argument('--im_sz', type=int, default=128, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--data_path', type=str, default='../dataset/gibson_room', help='path for dataset')
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
