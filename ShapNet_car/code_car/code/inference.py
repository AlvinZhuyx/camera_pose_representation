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
import transforms3d.euler as txe
import transforms3d.quaternions as txq

##########################################################################################
# build inference model upon learned vector representation
#########################################  data  #########################################
class dataloader(object):
    def __init__(self, path, nimg_per_ins, im_sz, validation_path=None, max_num_img=-1, max_ins=-1, test_path=None):
        assert max_num_img == -1 or max_num_img % 2 == 0
        assert nimg_per_ins % 2 == 0
        self.path = path
        self.validation_path = validation_path
        self.nimg_per_ins = nimg_per_ins
        instances = os.listdir(path)
        instances.sort()
        if test_path is not None:
            test_instance = os.listdir(test_path)
            test_instance.sort()
            self.test_path = test_path
            self.test_instance = test_instance
        else:
            self.test_path = None
            self.test_instance = None

        if max_ins > 0:
            self.test_path = path
            self.test_instance = instances[max_ins:]
            instances = instances[:max_ins]

        print('Our test path is: ', self.test_path)
        self.inst_dir = {}
        self.keys = instances
        self.im_sz = im_sz
        for item in instances:
            file = open(os.path.join(path, item, 'data_pair_infer_train.txt'))
            line = file.readline()
            valid_file = []
            while line:
                tmp_path0, tmp_name0 = line.split()
                valid_file.append((tmp_path0, tmp_name0))
                line = file.readline()
            file.close()
            if max_num_img > 0:
                valid_file = valid_file[:max_num_img]
            assert max_num_img < 0 or len(valid_file) == max_num_img
            self.inst_dir[item] = valid_file

    def get_test_item(self, idx):
        assert self.test_path is not None
        assert len(self.test_instance) > 0
        cur_ins = self.test_instance[idx]
        test_files = []
        file = open(os.path.join(self.test_path, cur_ins, 'data_pair_infer_test.txt'))
        line = file.readline()
        while line:
            tmp_path0, tmp_name0 = line.split()
            test_files.append((tmp_path0, tmp_name0))
            line = file.readline()
        file.close()
        test_imgs = []
        test_angles = []
        for path, name in test_files:
            image_path = os.path.join(path, 'rgb', name + '.png')
            angle_path = os.path.join(path, 'pose', name + '.txt')
            cur_img = load_rgb(image_path, self.im_sz)
            cur_angle = load_pose(angle_path)
            test_imgs.append(cur_img)
            pos = cur_angle[0:3, -1]
            q1 = txq.mat2quat(cur_angle[:3, :3])
            theta1_p, phi1_p, ksi1_p = txe.mat2euler(cur_angle[:3, :3], axes='rzxz')
            cur_r = np.sqrt(np.sum(pos * pos))
            tmpx = pos[0]
            tmpy = pos[1]
            tmpz = pos[2]
            theta = np.arctan2(tmpy, tmpx)
            phi = np.arccos(tmpz / cur_r)

            if theta < 0:
                theta += 2 * np.pi
            theta_aligned = theta - np.pi/2
            phi_aligned = np.pi - phi
            ksi_aligned = np.pi
            if np.abs(theta) < 2e-4 and np.abs(phi) < 2e-4:
				# the dataset represents camera pose for the top (view theta = 0 phi = 0) use a slightly different rotation matrix (corresponding to (-pi/2, pi, pi/2))
				# so here we just follow their definition in calculating the quaternion. Actually because we are calculate the angle difference between prediction and target,
                # so this does not influence our testing result.				
                ksi_aligned = np.pi / 2

            # this checks whether the result we get from aligned theta and phi agrees with the original rotation (directly from rotation matrix)
            # i.e. prove the correctness of our later on calculation for quaternion
            q2 = txe.euler2quat(theta_aligned, phi_aligned, ksi_aligned, axes='rzxz')
            d = np.abs(np.dot(q1, q2))
            d = min(1.0, max(-1.0, d))
            tmp_angle_diff = 2 * np.arccos(d) * 180 / np.pi
            if tmp_angle_diff > 1e-3:
                print(theta_aligned / np.pi * 180, phi_aligned / np.pi * 180, ksi_aligned / np.pi * 180)
                print(theta1_p / np.pi * 180, phi1_p / np.pi * 180, ksi1_p / np.pi * 180)
                print(tmp_angle_diff)
                assert False
            test_angles.append((theta, phi))

        test_imgs = np.array(test_imgs)
        test_angles = np.array(test_angles)
        return test_imgs, test_angles

    def get_num_ins(self):
        return len(self.keys)

    def get_validation_item(self, idx):
        assert self.validation_path is not None
        item = self.keys[idx]
        pose_path = os.path.join(os.path.join(self.validation_path, item), 'pose')
        image_path = os.path.join(os.path.join(self.validation_path, item), 'rgb')
        files = os.listdir(image_path)
        img_list = []
        angle_list = []
        for file in files:
            pose_name = os.path.join(pose_path, file[:-4] + '.txt')
            image_name = os.path.join(image_path, file)
            if not os.path.exists(pose_name):
                continue
            cur_img = load_rgb(image_name, self.im_sz)
            img_list.append(cur_img)
            cur_pose = load_pose(pose_name)
            pos = cur_pose[0:3, -1]
            cur_r = np.sqrt(np.sum(pos * pos))
            tmpx = pos[0]
            tmpy = pos[1]
            tmpz = pos[2]
            theta = np.arctan2(tmpy, tmpx)
            if theta < 0:
                theta += 2 * np.pi
            phi = np.arccos(tmpz / cur_r)
            angle_list.append((theta, phi))
        return np.array(img_list), np.array(angle_list)

    def get_item(self, idx, use_all=False):
        # randomly load nimg_per_ins images from the idx instance
        item = self.keys[idx]
        name_list = self.inst_dir[item]
        assert len(name_list) % 2 == 0
        if not use_all:
            chosen_idx = np.random.choice(len(name_list), size=self.nimg_per_ins, replace=False)
        else:
            chosen_idx = np.arange(len(name_list))
        img_list = []
        angle_list = []
        for idx in chosen_idx:
            path, name = name_list[idx]
            image_path = os.path.join(path, 'rgb', name + '.png')
            pose_path = os.path.join(path, 'pose', name + '.txt')
            cur_img = load_rgb(image_path, self.im_sz)
            img_list.append(cur_img)
            cur_pose = load_pose(pose_path)
            pos = cur_pose[0:3, -1]
            cur_r = np.sqrt(np.sum(pos * pos))
            tmpx = pos[0]
            tmpy = pos[1]
            tmpz = pos[2]
            theta = np.arctan2(tmpy, tmpx)
            if theta < 0:
                theta += 2 * np.pi
            phi = np.arccos(tmpz / cur_r)
            angle_list.append((theta, phi))
        img_list = np.array(img_list)
        angle_list = np.array(angle_list)
        if use_all:
            idx = np.argsort(angle_list[:, 0])
            tmp = img_list[idx]
            img_list = tmp
            angle_list = angle_list[idx]
            

        return img_list, angle_list


class recons_model(object):
    def __init__(self, FLAGS, sess):
        self.beta1 = FLAGS.beta1
        self.lr = FLAGS.lr
        self.im_sz = FLAGS.im_sz
        self.channel = FLAGS.channel
        self.epoch = FLAGS.epoch
        self.num_block = FLAGS.num_block
        self.block_size = FLAGS.block_size
        self.grid_num = FLAGS.grid_num
        self.grid_angle = FLAGS.grid_angle
        self.hidden_dim = self.num_block * self.block_size
        self.checkpoint_dir_load = FLAGS.checkpoint_dir_load
        self.checkpoint_dir_save = FLAGS.checkpoint_dir_save
        self.sess = sess
        self.nimg_per_ins = FLAGS.nimg_per_ins
        self.nins_per_batch = FLAGS.nins_per_batch
        self.print_iter = FLAGS.print_iter
        self.path = FLAGS.data_path
        self.val_path = FLAGS.validation_path
        self.test_path = FLAGS.test_path
        self.data_loader = dataloader(self.path, nimg_per_ins=self.nimg_per_ins, im_sz=self.im_sz, \
                                      validation_path=self.val_path, test_path=self.test_path)

    def inference_model(self, inputs, ins_idx, reuse=False):

        with tf.variable_scope('infer_model', reuse=reuse):
            # 128 * 128 --> 64 * 64
            num_ins = self.data_loader.get_num_ins()
            h1 = tf.layers.conv2d(inputs, 64, 4, 2, padding='same', name='conv1')
            h1 = tf.nn.leaky_relu(h1)
            # 64 * 64 -- > 32 * 32
            h2 = tf.layers.conv2d(h1, 64, 4, 2, padding='same', name='conv2')
            h2 = tf.nn.leaky_relu(h2)
            # 32 * 32 --> 16 * 16
            h3 = tf.layers.conv2d(h2, 128, 4, 2, padding='same', name='conv3')
            h3 = tf.nn.leaky_relu(h3)
            # 16 * 16 --> 8 * 8
            h4 = tf.layers.conv2d(h3, 128, 4, 2, padding='same', name='conv4')
            h4 = tf.nn.leaky_relu(h4)
            # 8 * 8 --> 4 * 4
            h5 = tf.layers.conv2d(h4, 256, 4, 2, padding='same', name='conv5')
            h5 = tf.nn.leaky_relu(h5)
            h5 = tf.reshape(h5, (-1, 256 * 4 * 4))

            # 4 * 4 * 512 --> 256 * 2 * 2
            h6_theta = tf.layers.dense(h5, self.hidden_dim, name='theta_head')
            h6_theta = h6_theta / tf.norm(h6_theta, axis=-1, keep_dims=True)

            h6_phi = tf.layers.dense(h5, self.hidden_dim, name='phi_head')
            h6_phi = h6_phi / tf.norm(h6_phi, axis=-1, keep_dims=True)
            return h6_theta, h6_phi

    def get_grid_code(self, v, B, grid_angle, rest_angle):
        grid_code = tf.gather(v, grid_angle, axis=0)
        a = rest_angle * self.grid_angle
        M_a = self.get_M(B, a)
        grid_code = self.motion_model(M_a, grid_code)
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
        grid_angle[:, 0] = np.clip(grid_angle[:, 0], 0, 2 * self.grid_num)
        grid_angle[:, 1] = np.clip(grid_angle[:, 1], 0, self.grid_num)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def build_model(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.im_sz, self.im_sz, self.channel])
        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[None])

        self.in_grid_theta = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_theta')
        self.in_rest_theta = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_theta')
        self.in_grid_phi = tf.placeholder(dtype=tf.int32, shape=[None], name='in_grid_phi')
        self.in_rest_phi = tf.placeholder(dtype=tf.float32, shape=[None], name='in_rest_phi')

        self.v_theta = tf.get_variable('Rotation_v_theta', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_phi = tf.get_variable('Rotation_v_phi', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.B_theta = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size,
                                                        name='Rotation_B_theta', antisym=True)
        self.B_phi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size,
                                                      name='Rotation_B_phi', antisym=True)

        self.v_theta_reg = self.v_theta / tf.norm(self.v_theta, axis=-1, keep_dims=True)
        self.v_phi_reg = self.v_phi / tf.norm(self.v_phi, axis=-1, keepdims=True)

        # predict vs underlying vector
        self.theta_pred, self.phi_pred = self.inference_model(self.inputs, self.in_ins_idx, reuse=False)
        self.theta_target = tf.stop_gradient(self.get_grid_code(self.v_theta_reg, self.B_theta, self.in_grid_theta, self.in_rest_theta))
        self.phi_target = tf.stop_gradient(self.get_grid_code(self.v_phi_reg, self.B_phi, self.in_grid_phi, self.in_rest_phi))
        self.view_loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.theta_target - self.theta_pred), axis=0)) + \
                         tf.reduce_sum(tf.reduce_mean(tf.square(self.phi_target - self.phi_pred), axis=0))

        update_var = []
        for var in tf.trainable_variables():
            if "infer_model" in var.name:
                print(var.name, var.shape)
                update_var.append(var)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9).minimize(self.view_loss, var_list=update_var)

    def save(self, step):
        model_name = 'infer.model'
        checkpoint_dir = self.checkpoint_dir_save
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

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
        from matplotlib import pyplot as plt
        self.build_model()
        load_var_list = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name:
                print('Load variable from generator: ', var.name)
                load_var_list.append(var)
        self.saver_load = tf.train.Saver(max_to_keep=1, var_list=load_var_list)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())
        print("Load rotation system from generator checkpoint")
        could_load, checkpoint_counter = self.load(self.checkpoint_dir_load, self.saver_load)
        assert could_load
        print("Try to load pretrained inference model")
        could_load, checkpoint_counter = self.load(self.checkpoint_dir_save, self.saver)
        assert could_load
        num_ins = self.data_loader.get_num_ins()

        # decoding system
        thetas = np.arange(0, 360, 0.25) / (180 / self.grid_num)
        phis = np.arange(0, 180, 0.25) / (180 / self.grid_num)
        grid_thetas = np.floor(thetas)
        rest_thetas = thetas - grid_thetas
        grid_phis = np.floor(phis)
        rest_phis = phis - grid_phis

        feed_dict = {
            self.in_grid_theta: grid_thetas,
            self.in_rest_theta: rest_thetas,
            self.in_grid_phi: grid_phis,
            self.in_rest_phi: rest_phis
        }
        v_theta_list, v_phi_list = self.sess.run([self.theta_target, self.phi_target], feed_dict=feed_dict)
        v_theta_list, v_phi_list = np.transpose(v_theta_list), np.transpose(v_phi_list)
        v_theta_norm = np.sum(np.square(v_theta_list), axis=0, keepdims=True)
        v_phi_norm = np.sum(np.square(v_phi_list), axis=0, keepdims=True)

        total_data = 0.0
        avg_angle_diff = 0.0
        start_time = time.time()
        count = 0
        b_sz = 200

        for i in range(num_ins): # should be num_ins
            imgs, angles = self.data_loader.get_test_item(i)
            angles /= self.grid_angle
            num_img = len(imgs)
            num_batch = math.ceil(float(num_img) / b_sz)

            for k in range(num_batch):
                start = k * b_sz
                end = min(num_img, (k + 1) * b_sz)
                cur_img = imgs[start: end]
                cur_angle = angles[start: end]
                cur_theta = cur_angle[:, 0]
                cur_phi = cur_angle[:, 1]

                cur_idx_in = np.array([i] * (end - start), dtype=np.int)
                feed_dict = {
                    self.inputs: cur_img,
                    self.in_ins_idx: cur_idx_in
                }

                v_theta_pred, v_phi_pred = self.sess.run([self.theta_pred, self.phi_pred], feed_dict=feed_dict)
                theta_vec_diff = -2 * np.matmul(v_theta_pred, v_theta_list) + v_theta_norm
                theta_idx = np.argmin(theta_vec_diff, axis=-1)
                theta_infer = thetas[theta_idx]
                phi_vec_diff = -2 * np.matmul(v_phi_pred, v_phi_list) + v_phi_norm
                phi_idx = np.argmin(phi_vec_diff, axis=-1)
                phi_infer = phis[phi_idx]

                angle_diff = []
                for jj in range(len(theta_infer)):
                    tmp_theta_infer = theta_infer[jj] * self.grid_angle
                    tmp_phi_infer = phi_infer[jj] * self.grid_angle
                    # transform to the camera pose rotation space (so that the quaternion agrees with the original rotation matrix)
                    theta_infer_aligned = tmp_theta_infer - np.pi / 2
                    phi_infer_aligned = np.pi - tmp_phi_infer
                    ksi_infer_aligned = np.pi
                    if np.abs(tmp_theta_infer) < 2e-4 and np.abs(tmp_phi_infer) < 2e-4:
                        ksi_infer_aligned = np.pi / 2
                    tmp_theta_target = cur_theta[jj] * self.grid_angle
                    tmp_phi_target = cur_phi[jj] * self.grid_angle
                    theta_target_aligned = tmp_theta_target - np.pi / 2
                    phi_target_aligned = np.pi - tmp_phi_target
                    ksi_target_aligned = np.pi
                    if np.abs(tmp_theta_target) < 2e-4 and np.abs(tmp_phi_target) < 2e-4:
                        ksi_target_aligned = np.pi / 2
                    q_infer = txe.euler2quat(theta_infer_aligned, phi_infer_aligned,ksi_infer_aligned, axes = 'rzxz')
                    q_target = txe.euler2quat(theta_target_aligned, phi_target_aligned, ksi_target_aligned, axes='rzxz')
                    d = np.abs(np.dot(q_infer, q_target))
                    d = min(1.0, max(-1.0, d))
                    tmp_angle_diff = 2 * np.arccos(d) * 180 / np.pi
                    angle_diff.append(tmp_angle_diff)
                angle_diff = np.array(angle_diff)
                avg_angle_diff += np.sum(angle_diff)
                total_data += len(angle_diff)

                if count % 100 == 0:
                    print("Epoch{}, ins{}, time {:.3f}, angle_diff {:.3f}".format(count, i, time.time() - start_time, avg_angle_diff/total_data))
                count += 1
        avg_angle_diff /= total_data
        print("final result angle diff {:.3f}".format(avg_angle_diff))

    def train(self):
        self.build_model()
        load_var_list = []
        for var in tf.trainable_variables():
            if 'Rotation' in var.name:
                print('Load variable from generator: ', var.name)
                load_var_list.append(var)
        self.saver_load = tf.train.Saver(max_to_keep=10, var_list=load_var_list)
        self.saver = tf.train.Saver(max_to_keep=10)
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
        avg_view_loss = 0.0

        # decoding system
        thetas = np.arange(0, 360, 1.0) / (180 / self.grid_num)
        phis = np.arange(0, 180, 1.0) / (180 / self.grid_num)
        grid_thetas = np.floor(thetas)
        rest_thetas = thetas - grid_thetas
        grid_phis = np.floor(phis)
        rest_phis = phis - grid_phis

        feed_dict = {
            self.in_grid_theta: grid_thetas,
            self.in_rest_theta: rest_thetas,
            self.in_grid_phi: grid_phis,
            self.in_rest_phi: rest_phis
        }
        v_theta_list, v_phi_list = self.sess.run([self.theta_target, self.phi_target], feed_dict=feed_dict)

        theta_corr = np.matmul(v_theta_list, np.transpose(v_theta_list))
        phi_corr = np.matmul(v_phi_list, np.transpose(v_phi_list))
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(theta_corr)
        plt.savefig(os.path.join('theta.png'))
        plt.close()
        plt.figure()
        plt.imshow(phi_corr)
        plt.savefig(os.path.join('phi.png'))
        plt.close()

        v_theta_list, v_phi_list = np.transpose(v_theta_list), np.transpose(v_phi_list)
        v_theta_norm = np.sum(np.square(v_theta_list), axis=0, keepdims=True)
        v_phi_norm = np.sum(np.square(v_phi_list), axis=0, keepdims=True)

        # load validation data
        valid_imgs = []
        valid_angles = []
        valid_idx = []
        for i in range(num_ins):
            cur_img, cur_angle = self.data_loader.get_validation_item(i)
            valid_imgs.append(cur_img)
            valid_angles.append(cur_angle)
            valid_idx.append(np.array([i] * len(cur_img)))

        valid_imgs = np.concatenate(valid_imgs, axis=0)
        valid_angles = np.concatenate(valid_angles, axis=0)
        valid_idx = np.concatenate(valid_idx, axis=0)
        valid_angles /= self.grid_angle
        assert len(valid_imgs) == len(valid_angles)
        assert len(valid_idx) == len(valid_imgs)
        num_valid = len(valid_imgs)
        valid_bsz = 200
        num_valid_batch = math.ceil(float(num_valid) / valid_bsz)


        for epoch in range(self.epoch):
            new_idx = np.random.permutation(num_ins)
            for i in range(n_batch):
                start = i * self.nins_per_batch
                end = min((i + 1) * self.nins_per_batch, num_ins)
                cur_ins_idx = new_idx[start: end].astype(np.int)

                cur_img = []
                cur_angle = []
                for idx in cur_ins_idx:
                    tmp_img, tmp_angle = self.data_loader.get_item(idx)
                    cur_img.append(tmp_img)
                    cur_angle.append(tmp_angle)

                cur_img = np.concatenate(cur_img, axis=0)
                cur_angle = np.concatenate(cur_angle, axis=0)
                cur_angle /= self.grid_angle
                cur_theta = cur_angle[:, 0]
                cur_phi = cur_angle[:, 1]
                cur_grid_theta = np.floor(cur_theta)
                cur_rest_theta = cur_theta - cur_grid_theta
                cur_grid_phi = np.floor(cur_phi)
                cur_rest_phi = cur_phi - cur_grid_phi

                in_idx = np.reshape(np.tile(np.expand_dims(cur_ins_idx, axis=-1), (1, self.nimg_per_ins)), (-1))
                feed_dict = {
                    self.inputs: cur_img,
                    self.in_ins_idx: in_idx,
                    self.in_grid_theta: cur_grid_theta,
                    self.in_rest_theta: cur_rest_theta,
                    self.in_grid_phi: cur_grid_phi,
                    self.in_rest_phi: cur_rest_phi
                }
                _, view_loss, v_theta_pred, v_phi_pred = self.sess.run([self.optim, self.view_loss, self.theta_pred, self.phi_pred], feed_dict=feed_dict)
                avg_view_loss += view_loss / self.print_iter

                theta_vec_diff = -2 * np.matmul(v_theta_pred, v_theta_list) + v_theta_norm
                theta_idx = np.argmin(theta_vec_diff, axis=-1)
                theta_infer = thetas[theta_idx]
                phi_vec_diff = -2 * np.matmul(v_phi_pred, v_phi_list) + v_phi_norm
                phi_idx = np.argmin(phi_vec_diff, axis=-1)
                phi_infer = phis[phi_idx]

                angle_diff = []
                for jj in range(len(theta_infer)):
                    tmp_theta_infer = theta_infer[jj] * self.grid_angle
                    tmp_phi_infer = phi_infer[jj] * self.grid_angle
                    # convert the angle of theta phi to the camera pose angle (defined by the camera pose matrix)
                    # the correctness of this conversion is checked in line101 - line104
                    theta_infer_aligned = tmp_theta_infer - np.pi / 2
                    phi_infer_aligned = np.pi - tmp_phi_infer
                    ksi_infer_aligned = np.pi
                    if np.abs(tmp_theta_infer) < 2e-4 and np.abs(tmp_phi_infer) < 2e-4:
                        ksi_infer_aligned = np.pi / 2
                    tmp_theta_target = cur_theta[jj] * self.grid_angle
                    tmp_phi_target = cur_phi[jj] * self.grid_angle
                    theta_target_aligned = tmp_theta_target - np.pi / 2
                    phi_target_aligned = np.pi - tmp_phi_target
                    ksi_target_aligned = np.pi
                    if np.abs(tmp_theta_target) < 2e-4 and np.abs(tmp_phi_target) < 2e-4:
                        ksi_target_aligned = np.pi / 2
                    q_infer = txe.euler2quat(theta_infer_aligned, phi_infer_aligned, ksi_infer_aligned, axes='rzxz')
                    q_target = txe.euler2quat(theta_target_aligned, phi_target_aligned, ksi_target_aligned, axes='rzxz')
                    d = np.abs(np.dot(q_infer, q_target))
                    d = min(1.0, max(-1.0, d))
                    tmp_angle_diff = 2 * np.arccos(d) * 180 / np.pi
                    angle_diff.append(tmp_angle_diff)
                angle_diff = np.array(angle_diff)

                if count % self.print_iter == 0:
                    print("Epoch{}, iter {}, time {:.3f}, avg view loss {:.3f}, angle_diff {:.3f}" \
                          .format(epoch, i, time.time() - start_time, avg_view_loss, np.mean(angle_diff)))
                    avg_view_loss = 0.0

                if count % (10 * self.print_iter) == 0:
                    v_start_time = time.time()
                    v_angle_diff = 0.0
                    total_num = 0.0
                    for j in range(num_valid_batch):
                        vstart = j * valid_bsz
                        vend = min((j + 1) * valid_bsz, num_valid)
                        cur_vimg = valid_imgs[vstart: vend]
                        cur_vangle = valid_angles[vstart: vend]
                        cur_vtheta = cur_vangle[:, 0]
                        cur_vphi = cur_vangle[:, 1]
                        cur_vidx = valid_idx[vstart: vend]
                        feed_dict = {
                            self.inputs: cur_vimg,
                            self.in_ins_idx: cur_vidx
                        }
                        v_theta_pred, v_phi_pred = self.sess.run([self.theta_pred, self.phi_pred], feed_dict=feed_dict)
                        theta_vec_diff = -2 * np.matmul(v_theta_pred, v_theta_list) + v_theta_norm
                        theta_idx = np.argmin(theta_vec_diff, axis=-1)
                        theta_infer = thetas[theta_idx]
                        phi_vec_diff = -2 * np.matmul(v_phi_pred, v_phi_list) + v_phi_norm
                        phi_idx = np.argmin(phi_vec_diff, axis=-1)
                        phi_infer = phis[phi_idx]

                        angle_diff = []
                        for jj in range(len(theta_infer)):
                            tmp_theta_infer = theta_infer[jj] * self.grid_angle
                            tmp_phi_infer = phi_infer[jj] * self.grid_angle
                            theta_infer_aligned = tmp_theta_infer - np.pi / 2
                            phi_infer_aligned = np.pi - tmp_phi_infer
                            ksi_infer_aligned = np.pi
                            if np.abs(tmp_theta_infer) < 2e-4 and np.abs(tmp_phi_infer) < 2e-4:
                                ksi_infer_aligned = np.pi / 2
                            tmp_theta_target = cur_theta[jj] * self.grid_angle
                            tmp_phi_target = cur_phi[jj] * self.grid_angle
                            theta_target_aligned = tmp_theta_target - np.pi / 2
                            phi_target_aligned = np.pi - tmp_phi_target
                            ksi_target_aligned = np.pi
                            if np.abs(tmp_theta_target) < 2e-4 and np.abs(tmp_phi_target) < 2e-4:
                                ksi_target_aligned = np.pi / 2
                            q_infer = txe.euler2quat(theta_infer_aligned, phi_infer_aligned, ksi_infer_aligned, axes='rzxz')
                            q_target = txe.euler2quat(theta_target_aligned, phi_target_aligned, ksi_target_aligned, axes='rzxz')
                            d = np.abs(np.dot(q_infer, q_target))
                            d = min(1.0, max(-1.0, d))
                            tmp_angle_diff = 2 * np.arccos(d) * 180 / np.pi
                            angle_diff.append(tmp_angle_diff)
                        angle_diff = np.array(angle_diff)
                        v_angle_diff += np.sum(angle_diff)
                        total_num += len(angle_diff)
                    assert total_num == num_valid
                    v_angle_diff /= num_valid
                    print('Validation result: time {:.3f} angle diff {:.3f}'.format(time.time() - v_start_time, v_angle_diff))

                count = count + 1

            if epoch % 25 == 0:
                self.save(epoch)


#########################################  config  #########################################

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs to train')
parser.add_argument('--nins_per_batch', type=int, default=10, help='Number of different instances in one batch')
parser.add_argument('--nimg_per_ins', type=int, default=20, help='Number of image chosen for one instance in one batch')
parser.add_argument('--print_iter', type=int, default=100, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')  # TODO was 0.003
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')
parser.add_argument('--train', type=bool, default=False, help='Train the model or test the trained model')

# structure parameters
parser.add_argument('--num_block', type=int, default=6, help='size of hidden dimension')
parser.add_argument('--block_size', type=int, default=16, help='size of hidden dimension')
parser.add_argument('--grid_num', type=int, default=18, help='Number of grid angle')
parser.add_argument('--grid_angle', type=float, default=np.pi/18, help='Size of one angle grid')
parser.add_argument('--im_sz', type=int, default=128, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--data_path', type=str, default='../../dataset/car/cars_train/', help='path for dataset')
parser.add_argument('--validation_path', type=str, default='../../dataset/car/cars_train_val/', help='path for validation samples')
parser.add_argument('--test_path', type=str, default='../../dataset/car/cars_train_test/', help='test path that include novel object')
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
