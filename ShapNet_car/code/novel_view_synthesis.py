from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os
from utils import *
from matplotlib import pyplot as plt
from matplotlib import cm
import argparse
import math

##########################################################################################
# use pair of data and small angle changes
# add addtional angle change to train rotation
# add position decoder
# replace bilinear interpolation with M rotation


#########################################  data  #########################################
# a dataloader that load part of the data at one time
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
            file = open(os.path.join(path, item, 'data_pair2_clean.txt'))
            line = file.readline()
            valid_file = []
            while line:
                tmp_path0, tmp_name0, tmp_path1, tmp_name1 = line.split()
                valid_file.append((tmp_path0, tmp_name0))
                valid_file.append((tmp_path1, tmp_name1))
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
        file = open(os.path.join(self.test_path, cur_ins, 'data_pair2.txt'))
        line = file.readline()
        while line:
            tmp_path0, tmp_name0 = line.split()
            test_files.append((tmp_path0, tmp_name0))
            line = file.readline()
        file.close()

        file = open(os.path.join(self.test_path, cur_ins, 'data_pair_extra2.txt'))
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
            cur_r = np.sqrt(np.sum(pos * pos))
            tmpx = pos[0]
            tmpy = pos[1]
            tmpz = pos[2]
            theta = np.arctan2(tmpy, tmpx)
            if theta < 0:
                theta += 2 * np.pi
            phi = np.arccos(tmpz / cur_r)
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
            tmp_idx = np.random.choice(len(name_list) // 2, size=self.nimg_per_ins // 2, replace=False)
            chosen_idx = []
            for idx in tmp_idx:
                chosen_idx.append(2 * idx)
                chosen_idx.append(2 * idx + 1)
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
            idx = np.argsort(angle_list[:,0])
            tmp = img_list[idx]
            img_list = tmp
            angle_list = angle_list[idx]
            

        return img_list, angle_list

######################################### model ##########################################
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
        self.max_ins = FLAGS.max_ins
        self.print_iter = FLAGS.print_iter
        self.grid_angle = FLAGS.grid_angle
        self.grid_num = FLAGS.grid_num
        self.update_step = FLAGS.update_step
        self.update_step_sz = FLAGS.update_step_sz
        self.path = FLAGS.data_path
        self.validation_path = FLAGS.validation_path

        assert self.nimg_per_ins % 2 == 0
        self.npair_per_ins = self.nimg_per_ins // 2
        self.data_loader = dataloader(FLAGS.data_path, nimg_per_ins=self.nimg_per_ins, im_sz=self.im_sz, validation_path=FLAGS.validation_path, max_num_img=FLAGS.max_num_img, max_ins=FLAGS.max_ins, test_path=FLAGS.test_path)

    def decoder(self, input, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            h1 = tf.layers.dense(input, 4 * 4 * 256, name='h1')
            h1 = tf.nn.leaky_relu(h1)
            # 4
            h1 = tf.reshape(h1, (-1, 4, 4, 256))
            # 8
            h2 = tf.layers.conv2d_transpose(h1, 256, 4, 2, padding='same', name='h2')
            h2 = tf.nn.leaky_relu(h2)
            # 16
            h3 = tf.keras.layers.UpSampling2D((2, 2))(h2)
            h4 = tf.layers.conv2d_transpose(h3, 256, 4, 1, padding='same', name='h4')
            h4 = tf.nn.leaky_relu(h4) + h3
            # 32
            h5 = tf.keras.layers.UpSampling2D((2, 2))(h4)
            h6 = tf.layers.conv2d_transpose(h5, 128, 4, 1, padding='same', name='h6')
            h6 = tf.nn.leaky_relu(h6)
            # 64
            h7 = tf.keras.layers.UpSampling2D((2, 2))(h6)
            h8 = tf.layers.conv2d_transpose(h7, 64, 4, 1, padding='same', name='h8')
            h8 = tf.nn.leaky_relu(h8)
            # 128
            h9 = tf.keras.layers.UpSampling2D((2, 2))(h8)
            h10 = tf.layers.conv2d_transpose(h9, 3, 4, 1, padding='same', name='h10')

        return tf.nn.tanh(h10)

    def build_model(self):
        self.in_image0 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, self.im_sz, self.im_sz, self.channel])
        self.in_image1 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, self.im_sz, self.im_sz, self.channel])
        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins])
        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_a0_grid')
        self.in_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_a1_grid')
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_a0_rest')
        self.in_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='in_a1_rest')
        

        self.add_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='add_a0')
        self.add_grid_angle1 = tf.placeholder(dtype=tf.int32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='add_a1')
        self.add_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='add_a0_rest')
        self.add_rest_angle1 = tf.placeholder(dtype=tf.float32, shape=[self.nins_per_batch * self.npair_per_ins, 2], name='add_a1_rest')
       
        
        num_ins = self.data_loader.get_num_ins()
        self.v_instance = tf.get_variable('v_instance', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
        self.v_theta = tf.get_variable('Rotation_v_theta', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_phi = tf.get_variable('Rotation_v_phi', initializer=tf.convert_to_tensor(np.random.normal(scale=0.001, size=[self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.B_theta = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_theta', antisym=True)
        self.B_phi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_phi', antisym=True)
        

        self.v_ins_reg = self.v_instance / tf.norm(self.v_instance, axis=-1, keep_dims=True)
        self.v_theta_reg = self.v_theta / tf.norm(self.v_theta, axis=-1, keep_dims=True)
        self.v_phi_reg = self.v_phi / tf.norm(self.v_phi, axis=-1, keepdims=True)
        
        v_theta0 = self.get_grid_code(self.v_theta_reg, self.B_theta, self.in_grid_angle0[:, 0], self.in_rest_angle0[:, 0])
        v_theta1 = self.get_grid_code(self.v_theta_reg, self.B_theta,self.in_grid_angle1[:, 0], self.in_rest_angle1[:, 0])
        v_theta_add0 = self.get_grid_code(self.v_theta_reg, self.B_theta, self.add_grid_angle0[:, 0], self.add_rest_angle0[:, 0])
        v_theta_add1 = self.get_grid_code(self.v_theta_reg, self.B_theta, self.add_grid_angle1[:, 0], self.add_rest_angle1[:, 0])

        v_phi0 = self.get_grid_code(self.v_phi_reg, self.B_phi, self.in_grid_angle0[:, 1], self.in_rest_angle0[:, 1])
        v_phi1 = self.get_grid_code(self.v_phi_reg, self.B_phi, self.in_grid_angle1[:, 1], self.in_rest_angle1[:, 1])
        v_phi_add0 = self.get_grid_code(self.v_phi_reg, self.B_phi, self.add_grid_angle0[:, 1], self.add_rest_angle0[:, 1])
        v_phi_add1 = self.get_grid_code(self.v_phi_reg, self.B_phi, self.add_grid_angle1[:, 1], self.add_rest_angle1[:, 1])

        v_ins = tf.gather(self.v_ins_reg, self.in_ins_idx)
        v_total0 = tf.concat([v_ins, v_theta0, v_phi0], axis=-1)
        v_total1 = tf.concat([v_ins, v_theta1, v_phi1], axis=-1)
        
        # reconstruction loss
        self.re_img0 = self.decoder(v_total0, reuse=False)
        self.re_img1 = self.decoder(v_total1, reuse=True)
        self.recons_loss = 0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img0 - self.in_image0), axis=0)) +\
                           0.5 * tf.reduce_sum(tf.reduce_mean(tf.square(self.re_img1 - self.in_image1), axis=0))
                           
        # rotation loss
        delta_angle = (tf.cast(self.in_grid_angle1, tf.float32) + self.in_rest_angle1 - tf.cast(self.in_grid_angle0, tf.float32) - self.in_rest_angle0) * self.grid_angle
        delta_angle_add = (tf.cast(self.add_grid_angle1, tf.float32) + self.add_rest_angle1 - tf.cast(self.add_grid_angle0, tf.float32) - self.add_rest_angle0) * self.grid_angle
        theta = delta_angle[:, 0]
        phi = delta_angle[:, 1]
        M_theta = self.get_M(self.B_theta, theta)
        M_phi = self.get_M(self.B_phi, phi)
        v_theta0_rot = self.motion_model(M_theta, v_theta0)
        v_phi0_rot = self.motion_model(M_phi, v_phi0)

        theta_add = delta_angle_add[:, 0]
        phi_add = delta_angle_add[:, 1]
        M_theta_add = self.get_M(self.B_theta, theta_add)
        M_phi_add = self.get_M(self.B_phi, phi_add)
        v_theta_add0_rot = self.motion_model(M_theta_add, v_theta_add0)
        v_phi_add0_rot = self.motion_model(M_phi_add, v_phi_add0)
        
        self.rot_loss = 0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_theta0_rot - v_theta1), axis=0)) + \
                        0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_theta_add0_rot - v_theta_add1), axis=0)) + \
                        0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_phi0_rot - v_phi1), axis=0)) + \
                        0.5 * self.rot_reg_weight * tf.reduce_sum(tf.reduce_mean(tf.square(v_phi_add0_rot - v_phi_add1), axis=0))
        
        rot_var = []
        dec_var = []
        B_var = []
        for var in tf.trainable_variables():
            if 'Rotation_B' in var.name:
                B_var.append(var)
            if 'Rotation' in var.name:
                print('Rotaion variable: ', var.name)
                rot_var.append(var)
            else:
                print('Decoder variable: ', var.name)
                dec_var.append(var)

        self.update_dec = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.recons_loss, var_list=dec_var)
        self.update_rot = tf.train.AdamOptimizer(self.update_step_sz, beta1=self.beta1).minimize(self.recons_weight * self.recons_loss + self.rot_loss, var_list=rot_var)

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
        #assert np.max(angle[:, 0]) <= 2 * self.grid_num and np.min(angle[:, 0]) >= 0
        #assert np.max(angle[:, 1]) <= self.grid_num and np.min(angle[:, 0]) >= 0
        grid_angle = np.floor(angle) + np.random.randint(low=0, high=2, size=angle.shape)
        grid_angle[:, 0] = np.clip(grid_angle[:, 0], 0, 2 * self.grid_num)
        grid_angle[:, 1] = np.clip(grid_angle[:, 1], 0, self.grid_num)
        rest_angle = angle - grid_angle.astype(np.float32)
        return grid_angle.astype(np.int32), rest_angle.astype(np.float32)

    def validate(self, count):
        assert self.validation_path is not None
        num_ins = self.data_loader.get_num_ins()

        batch_size = self.nins_per_batch * self.npair_per_ins
        img_queue = None
        angle_queue = None
        psnr_list = []
        loss_list = []
        idx_list = None
        flag = True
        
        idxs = np.random.permutation(num_ins)
        for i in range(num_ins):
            idx = idxs[i]
            tmp_img, tmp_angle = self.data_loader.get_validation_item(idx)
            if img_queue is None:
                img_queue = np.copy(tmp_img)
                angle_queue = np.copy(tmp_angle)
                idx_list = np.array([idx] * len(tmp_img))
            else:
                img_queue = np.append(img_queue, tmp_img, axis=0)
                angle_queue = np.append(angle_queue, tmp_angle, axis=0)
                idx_list = np.append(idx_list, np.array([idx] * len(tmp_img)), axis=0)
            if len(img_queue) >= batch_size:

                img_in = np.copy(img_queue[0: batch_size])
                angle_in = np.copy(angle_queue[0: batch_size]) / self.grid_angle
                idx_in = np.copy(idx_list[0 : batch_size])
                if len(img_queue) == batch_size:
                    img_queue = None
                    angle_queue = None
                    idx_list = None
                else:
                    img_queue = np.copy(img_queue[batch_size: ])
                    angle_queue = np.copy(angle_queue[batch_size: ])
                    idx_list = np.copy(idx_list[batch_size: ])
                    
                
                grid_angle_in, rest_angle_in = self.quantize_angle(angle_in)
                feed_dict = {self.in_ins_idx: idx_in,
                             self.in_grid_angle0: grid_angle_in,
                             self.in_rest_angle0: rest_angle_in}
                    
                re_img = self.sess.run(self.re_img0, feed_dict=feed_dict)
                recons_psnr = calculate_psnr(re_img, img_in)
                psnr_list.append(recons_psnr)
                loss_list.append(np.sum(np.square(re_img - img_in), axis=(1, 2, 3)))
                if flag:
                    save_img_recons = np.copy(re_img)
                    save_img_ori = np.copy(img_in)
                    flag = False
                
        
        if len(img_queue) > 0:
            angle_in = np.zeros((batch_size, 2))
            idx_in = np.zeros(batch_size, dtype=np.int32)
            angle_in[: len(img_queue)] = angle_queue / self.grid_angle
            idx_in[: len(img_queue)] = idx_list
            grid_angle_in, rest_angle_in = self.quantize_angle(angle_in)
            feed_dict = {self.in_ins_idx: idx_in,
                         self.in_grid_angle0: grid_angle_in,
                         self.in_rest_angle0: rest_angle_in}
            re_img = self.sess.run(self.re_img0, feed_dict=feed_dict)
            recons_psnr = calculate_psnr(re_img[: len(img_queue)], img_queue)
            psnr_list.append(recons_psnr)
            loss_list.append(np.sum(np.square(re_img[: len(img_queue)] - img_queue), axis=(1, 2, 3)))
        psnr_list = np.concatenate(psnr_list, axis=0)
        loss_list = np.concatenate(loss_list, axis=0)
        print('psnr0 mean/max/min {:.2f}/{:.2f}/{:.2f}; recons loss0 mean/max/min {:.2f}/{:.2f}/{:.2f};'.\
              format(np.mean(psnr_list), np.max(psnr_list), np.min(psnr_list), np.mean(loss_list), np.max(loss_list), np.min(loss_list)))
            

        save_images(save_img_recons[: 64], os.path.join(self.sample_dir, 'valid_{:06d}.png'.format(count)))
        save_images(save_img_ori[: 64], os.path.join(self.sample_dir, 'valid_{:06d}_ori.png'.format(count)))

    def print_loc(self, save_path='v_vis', count=0):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res = self.sess.run([self.v_theta_reg, self.v_phi_reg])
        name = ['theta', 'phi']
        for i in range(2):
            v = res[i]
            cor = np.matmul(v, np.transpose(v))
            plt.figure()
            plt.imshow(cor)
            plt.savefig(os.path.join(save_path,  '{}_{:06d}.png'.format(name[i], count)))
            plt.close()

    def test_noise(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        num_ins = len(self.data_loader.test_instance)

        self.in_ins_idx = tf.placeholder(dtype=tf.int32, shape=[None])
        # theta1 and theta0 should be converted into grid space
        self.in_grid_angle0 = tf.placeholder(dtype=tf.int32, shape=[None, 2])
        self.in_rest_angle0 = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.in_theta_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_dim])
        self.in_phi_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_dim])

        self.v_instance = tf.get_variable('v_instance', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[num_ins, 128]), dtype=tf.float32))
        self.v_theta = tf.get_variable('Rotation_v_theta', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[2 * self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.v_phi = tf.get_variable('Rotation_v_phi', initializer=tf.convert_to_tensor(
            np.random.normal(scale=0.001, size=[self.grid_num + 1, self.hidden_dim]), dtype=tf.float32))
        self.B_theta = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_theta', antisym=True)
        self.B_phi = construct_block_diagonal_weights(num_block=self.num_block, block_size=self.block_size, name='Rotation_B_phi', antisym=True)
        self.v_ins_reg = self.v_instance / tf.norm(self.v_instance, axis=-1, keep_dims=True)
        self.v_theta_reg = self.v_theta / tf.norm(self.v_theta, axis=-1, keep_dims=True)
        self.v_phi_reg = self.v_phi / tf.norm(self.v_phi, axis=-1, keepdims=True)

        v_theta0 = self.get_grid_code(self.v_theta_reg, self.B_theta, self.in_grid_angle0[:, 0], self.in_rest_angle0[:, 0])
        v_phi0 = self.get_grid_code(self.v_phi_reg, self.B_phi, self.in_grid_angle0[:, 1], self.in_rest_angle0[:, 1])

        v_theta0_noise = v_theta0 + self.in_theta_noise
        v_phi0_noise = v_phi0 + self.in_phi_noise
        v_ins = tf.gather(self.v_ins_reg, self.in_ins_idx)
        v_total0 = tf.concat([v_ins, v_theta0_noise, v_phi0_noise], axis=-1)

        self.re_img0 = self.decoder(v_total0, reuse=False)
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load()
        assert could_load

        
        # calculate the std of each position of our view vector
        theta_vectors, phi_vectors = self.sess.run([self.v_theta_reg, self.v_phi_reg])
        theta_vectors = np.reshape(theta_vectors, ((2*self.grid_num + 1), self.hidden_dim))
        phi_vectors = np.reshape(phi_vectors, (self.grid_num + 1, self.hidden_dim))
        theta_std = np.std(theta_vectors, axis=0)
        phi_std = np.std(phi_vectors, axis=0)
        avg_loss = {}
        avg_psnr = {}
        start_time = time.time()
        for i in range(num_ins):
            imgs, angles = self.data_loader.get_test_item(i)
            angles = angles / self.grid_angle
            idx = np.array([i] * len(imgs), dtype=np.int32)
            grid_angles, rest_angles = self.quantize_angle(angles)
            for j in range(13):
                cur_theta_std = 0.5 / 20.0 * j * theta_std
                cur_phi_std = 0.5 / 20.0 * j * phi_std
                cur_theta_noise = np.expand_dims(cur_theta_std, axis=0) * np.random.normal(loc=0.0, scale=1.0, size=(len(grid_angles), self.hidden_dim))
                cur_phi_noise = np.expand_dims(cur_phi_std, axis=0) * np.random.normal(loc=0.0, scale=1.0, size=(len(grid_angles), self.hidden_dim))

                feed_dict = {
                    self.in_ins_idx: idx,
                    self.in_grid_angle0: grid_angles,
                    self.in_rest_angle0: rest_angles,
                    self.in_theta_noise: cur_theta_noise,
                    self.in_phi_noise: cur_phi_noise
                }
                re_imgs = self.sess.run(self.re_img0, feed_dict=feed_dict)
                if i < 30:
                    save_images(imgs[:64], os.path.join(save_path, '{}_{}_ori.png'.format(i, j)))
                    save_images(re_imgs[:64], os.path.join(save_path, '{}_{}_recons.png'.format(i, j)))
                avg_loss[(i, j)] = np.sum(np.mean(np.square(re_imgs - imgs), axis=0))
                avg_psnr[(i, j)] = np.mean(calculate_psnr(re_imgs, imgs))
                print("ins {} time {} noise mag {} loss {} psnr {}".format(i, time.time() - start_time, 0.5/20*j, avg_loss[(i, j)], avg_psnr[(i, j)]))
        
        for j in range(13):
            total_avg_loss = 0.0
            total_avg_psnr = 0.0
            for i in range(num_ins):
                total_avg_loss += avg_loss[(i, j)] / float(num_ins)
                total_avg_psnr += avg_psnr[(i, j)] / float(num_ins)
            print("total average noise mag {} loss {} psnr {}".format(0.5/20*j, total_avg_loss, total_avg_psnr))

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
        avg_rot_loss = 0.0
        avg_recons_loss = 0.0


        for epoch in range(self.epoch):
            new_idx = np.random.permutation(num_ins)
            for i in range(num_iteration):
                start = i * self.nins_per_batch
                end = min((i+1) * self.nins_per_batch, num_ins)
                cur_ins_idx = new_idx[start: end].astype(np.int)

                cur_img0 = []
                cur_img1 = []
                cur_angle0 = []
                cur_angle1 = []
                
                for idx in cur_ins_idx:
                    tmp_img, tmp_angle = self.data_loader.get_item(idx)
                    tmp_img = np.reshape(tmp_img, (self.npair_per_ins, 2, self.im_sz, self.im_sz, self.channel))
                    tmp_angle /= self.grid_angle
                    tmp_angle = np.reshape(tmp_angle, (self.npair_per_ins, 2, 2))
                    cur_img0.append(tmp_img[:, 0])
                    cur_img1.append(tmp_img[:, 1])
                    cur_angle0.append(tmp_angle[:, 0])
                    cur_angle1.append(tmp_angle[:, 1])

                
                cur_img0 = np.concatenate(cur_img0, axis = 0)
                cur_img1 = np.concatenate(cur_img1, axis = 0)
                cur_angle0 = np.concatenate(cur_angle0, axis = 0)
                cur_angle1 = np.concatenate(cur_angle1, axis = 0)
                assert np.all(np.abs(cur_angle0 - cur_angle1) < 1.5)
                cur_ins_idx = np.reshape(np.tile(np.expand_dims(cur_ins_idx, axis=-1), (1, self.npair_per_ins)), (-1))

                add_theta0 = np.random.uniform(low=0.0, high=2 * self.grid_num - 1.0, size=(self.nins_per_batch * self.npair_per_ins , 1))
                add_theta1 = add_theta0 + np.random.uniform(low=0.5, high=1.0, size=(self.nins_per_batch * self.npair_per_ins , 1))
                add_theta1 = np.clip(add_theta1, 0.0, 2 * self.grid_num)
                add_phi0 = np.random.uniform(low=0.0, high=self.grid_num - 1.0, size=(self.nins_per_batch * self.npair_per_ins , 1))
                add_phi1 = add_phi0 + np.random.uniform(low=0.5, high=1.0, size=(self.nins_per_batch * self.npair_per_ins , 1))
                add_phi1 = np.clip(add_phi1, 0.0, self.grid_num)
                
                add_angle0 = np.concatenate([add_theta0, add_phi0], axis=-1)
                add_angle1 = np.concatenate([add_theta1, add_phi1], axis=-1)
                
                
                cur_grid_angle0, cur_rest_angle0 = self.quantize_angle(cur_angle0)
                cur_grid_angle1, cur_rest_angle1 = self.quantize_angle(cur_angle1)
                add_grid_angle0, add_rest_angle0 = self.quantize_angle(add_angle0)
                add_grid_angle1, add_rest_angle1 = self.quantize_angle(add_angle1)
                
                
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
                             self.add_rest_angle1: add_rest_angle1}

                for _ in range(self.update_step):
                    self.sess.run(self.update_rot, feed_dict=feed_dict)

                res = self.sess.run([self.update_dec, self.rot_loss, self.recons_loss, self.re_img0, self.re_img1], feed_dict=feed_dict)
                _, rot_loss, recons_loss, re_img0, re_img1 = res

                re_img = np.concatenate([np.expand_dims(re_img0, axis = 1), np.expand_dims(re_img1, axis = 1)], axis = 1)
                re_img = np.reshape(re_img, (self.nins_per_batch * self.nimg_per_ins, self.im_sz, self.im_sz, self.channel))
                cur_img = np.concatenate([np.expand_dims(cur_img0, axis = 1), np.expand_dims(cur_img1, axis = 1)], axis = 1)
                cur_img = np.reshape(cur_img, (self.nins_per_batch * self.nimg_per_ins, self.im_sz, self.im_sz, self.channel))
                recons_psnr = np.mean(calculate_psnr(re_img, cur_img))
                avg_rot_loss += rot_loss / self.print_iter
                avg_recons_loss += recons_loss / self.print_iter

                if count % self.print_iter == 0:
                    print('Epoch {} iter {} time {:.3f} rotation loss {:.3f} recons loss {:.3f} psnr {:.2f}'.\
                          format(epoch, i, time.time() - start_time, avg_rot_loss, avg_recons_loss, recons_psnr))

                    avg_rot_loss = 0.0
                    avg_recons_loss = 0.0


                if count % (self.print_iter * 1) == 0:
                    save_images(cur_img[:64], os.path.join(self.sample_dir, 'ori{:06d}.png'.format(count)))
                    save_images(re_img[:64], os.path.join(self.sample_dir, 'recons0_img{:06d}.png'.format(count)))

                if count % (self.print_iter * 3) == 0: #20
                    self.validate(count)
                    self.print_loc(self.sample_dir, count)

                if count % (self.print_iter * 6) == 0: #40
                    self.save(count)

                count = count + 1

#########################################  config  #########################################

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--epoch', type=int, default=1500, help='Number of epochs to train')
parser.add_argument('--nins_per_batch', type=int, default=20, help='Number of different instances in one batch')
parser.add_argument('--nimg_per_ins', type=int, default=10, help='Number of image chosen for one instance in one batch, since we use pair of data, should be multiple of 2')
parser.add_argument('--print_iter', type=int, default=1070, help='Number of iteration between print out')
parser.add_argument('--lr', type=float, default=1.0e-4, help='Learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')
parser.add_argument('--update_step', type=int, default=3, help='Number of inference step in Langevin')
parser.add_argument('--update_step_sz', type=float, default=1.0e-4, help='Step size for Langevin update')
parser.add_argument('--train', type=bool, default=False, help='train the model or test robustness to noise')
# weight of different losses
parser.add_argument('--recons_weight', type=float, default=0.05, help='Reconstruction loss weight')
parser.add_argument('--rot_reg_weight', type=float, default=50.0, help='Regularization weight for whether vectors agree with each other')
# structure parameters
parser.add_argument('--num_block', type=int, default=6, help='Number of blocks in the representation')
parser.add_argument('--block_size', type=int, default=16, help='Number of neurons per block')
parser.add_argument('--grid_num', type=int, default=18, help='number of grid angle')
parser.add_argument('--grid_angle', type=float, default=np.pi / 18., help='size of one angle grid')
# dataset parameters
parser.add_argument('--im_sz', type=int, default=128, help='size of image')
parser.add_argument('--channel', type=int, default=3, help='channel of image')
parser.add_argument('--max_ins', type=int, default=-1, help='number of instance used in training, set -1 will use all')
parser.add_argument('--max_num_img', type=int, default=-1, help='number of images used per instance, set -1 will use all')
parser.add_argument('--data_path', type=str, default='../dataset/car/cars_train/', help='path for dataset')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='path for saving checkpoint')
parser.add_argument('--sample_dir', type=str, default='sample', help='path for save samples')
parser.add_argument('--validation_path', type=str, default='../dataset/car/cars_train_val/', help='path for validation samples')
parser.add_argument('--test_path', type=str, default='../dataset/car/cars_train_test/', help='test path that include novel object')

FLAGS = parser.parse_args()
def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.set_random_seed(1234)
    np.random.seed(2345)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    with tf.Session() as sess:
        model = recons_model(FLAGS, sess)
        if FLAGS.train:
            model.train()
        else:
            model.test_noise('test_noise')

if __name__ == '__main__':
    tf.app.run()



