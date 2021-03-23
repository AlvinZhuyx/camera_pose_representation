from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy.stats import norm
import sys
import cv2
import imageio
import skimage
import skimage.measure as measure
from PIL import Image

def calculate_psnr(pred, truth):
    assert len(pred) == len(truth)
    n = len(pred)
    psnr_list = []
    for i in range(n):
        cur_pred = pred[i]
        cur_truth = truth[i]
        cur_pred = cur_pred / 2. + 0.5
        cur_pred = np.clip(cur_pred, 0.0, 1.0)
        cur_truth = cur_truth / 2. + 0.5
        cur_truth = np.clip(cur_truth, 0.0, 1.0)
        psnr = measure.compare_psnr(cur_truth, cur_pred, data_range=1)
        psnr_list.append(psnr)
    return psnr_list

def merge_images(images, space=0, mean_img=None):
    num_images = images.shape[0]
    canvas_size = int(np.ceil(np.sqrt(num_images)))
    h = images.shape[1]
    w = images.shape[2]
    canvas = np.zeros((canvas_size * h + (canvas_size-1) * space,  canvas_size * w + (canvas_size-1) * space, 3), np.uint8)

    for idx in range(num_images):
        image = images[idx,:,:,:]
        if mean_img:
            image += mean_img
        i = idx % canvas_size
        j = idx // canvas_size
        min_val = np.min(image)
        max_val = np.max(image)
        image = ((image - min_val) / (max_val - min_val + 1e-6) * 255).astype(np.uint8)
        canvas[j*(h+space):j*(h+space)+h, i*(w+space):i*(w+space)+w,:] = image
    return canvas

def save_images(images, file_name, space=0, mean_img=None):
    cell_image = merge_images(images, space, mean_img)
    img = Image.fromarray(np.uint8(cell_image))
    img.save(file_name)

def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()

def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    return img

def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img

def construct_block_diagonal_weights(num_channel, num_block, block_size, name='M', antisym=False):
    if not antisym:
        M = tf.Variable(tf.random_normal([num_block, num_channel, block_size, block_size], stddev=0.001), name=name)
    else:
        M = tf.Variable(tf.random_normal([num_block, num_channel, int(block_size * (block_size-1) / 2)], stddev=0.001), name=name)
        M = construct_antisym_matrix(M, block_size)
    M = block_diagonal(M)

    return M

def construct_antisym_matrix(value, dim):
    assert value.shape.as_list()[-1] == dim * (dim - 1) / 2
    batch_shape = value.shape.as_list()[:-1]
    ones = tf.ones((dim, dim), dtype=tf.int64) #size of the output matrix
    mask_a = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.subtract(mask_a, mask_b) # Mask of upper triangle above diagonal

    zero = tf.constant(0, dtype=tf.int64)
    non_zero = tf.not_equal(mask, zero) #Conversion of mask to Boolean matrix
    non_zero = tf.tile(tf.reshape(non_zero, [1] * len(batch_shape) + non_zero._shape_as_list()), batch_shape + [1] * len(non_zero._shape_as_list()))

    indices = tf.where(non_zero) # Extracting the indices of upper trainagle elements

    out = tf.SparseTensor(indices, tf.reshape(value, [-1]), dense_shape=batch_shape + [dim, dim])
    dense = tf.sparse_tensor_to_dense(out)

    return dense - tf.transpose(dense, [*range(len(batch_shape))] + [len(batch_shape)+1, len(batch_shape)])

def block_diagonal(matrices):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
      matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
      dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
      A matrix with the input matrices stacked along its main diagonal, having
      shape [..., \sum_i N_i, \sum_i M_i].

    """
    # matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)

    if type(matrices) is not list:
        matrices = tf.unstack(matrices)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked