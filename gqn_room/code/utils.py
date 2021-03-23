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
import matplotlib.cm as cm
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

def AdaIn(features, scale, bias):
    """
    Adaptive instance normalization component. Works with both 4D and 5D tensors
    :features: features to be normalized
    :scale: scaling factor. This would otherwise be calculated as the sigma from a "style" features in style transfer
    :bias: bias factor. This would otherwise be calculated as the mean from a "style" features in style transfer
    """

    mean, variance = tf.nn.moments(features, list(range(len(features.get_shape())))[1:-1],
                                   keep_dims=True)  # Only consider spatial dimension
    sigma = tf.rsqrt(variance + 1e-8)
    normalized = (features - mean) * sigma
    scale_broadcast = tf.reshape(scale, tf.shape(mean))
    bias_broadcast = tf.reshape(bias, tf.shape(mean))
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized

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


def draw_heatmap(data, save_path, xlabels=None, ylabels=None):
    # data = np.clip(data, -0.05, 0.05)
    cmap = cm.get_cmap('rainbow', 1000)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)

    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    plt.savefig(save_path)
    plt.close()


def shape_mask(size, shape):
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    if shape == 'square':
        mask = np.ones_like(x, dtype=bool)
    elif shape == 'circle':
        mask = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) <= 0.5
    elif shape == 'triangle':
        mask = (y + 2 * x >= 1) * (-y + 2 * x <= 1)
    else:
        raise NotImplementedError

    return mask


def draw_heatmap_2D(data, vmin=None, vmax=None, shape='square'):
    place_size, _ = np.shape(data)
    place_mask = shape_mask(place_size, shape)
    if vmin is None:
        vmin = data[place_mask].min() - 0.01
    if vmax is None:
        vmax = data[place_mask].max()
    data[~place_mask] = vmin - 1

    cmap = cm.get_cmap('rainbow', 1000)
    cmap.set_under('w')
    plt.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')


def draw_path_to_target(place_len, place_seq, save_file=None, target=None, obstacle=None, a=None, b=None, col_scheme='single'):
    if save_file is not None:
        plt.figure(figsize=(5, 5))
    if type(place_seq) == list or np.ndim(place_seq) > 2:
        if col_scheme == 'rainbow':
            colors = cm.rainbow(np.linspace(0, 1, len(place_seq)))
        for i in range(len(place_seq)):
            p_seq = place_seq[i]
            color = colors[i] if col_scheme == 'rainbow' else 'darkcyan'
            label = 'a=%.2f, b=%d' % (a[i], b[i]) if (a is not None and b is not None) else ''
            plt.plot(p_seq[:, 1], place_len - p_seq[:, 0], 'o-', color=color,
                     ms=6, lw=2.0, label=label)
            if a is not None and len(a) > 1:
                plt.legend(loc='lower left', fontsize=12)
    else:
        if type(place_seq) == list:
            place_seq = place_seq[0]
        plt.plot(place_seq[:, 1], place_len - place_seq[:, 0], 'o-', ms=6, lw=2.0, color='darkcyan')

    if target is not None:
        plt.plot(target[1], place_len - target[0], '*', ms=12, color='r')
        if obstacle is not None:
            if np.ndim(obstacle) == 2:
                obstacle = obstacle.T
            plt.plot(obstacle[1], place_len - obstacle[0], 's', ms=8, color='dimgray')

    plt.xticks([])
    plt.yticks([])
    plt.xlim((0, place_len))
    plt.ylim((0, place_len))
    ax = plt.gca()
    ax.set_aspect(1)
    if save_file is not None:
        plt.savefig(save_file)
        plt.close()

# def draw_path_to_target(place_len, place_seq, target=None, obstacle=None, col=(255, 0, 0)):
#     place_seq = np.round(place_seq).astype(int)
#     cmap = cm.get_cmap('rainbow', 1000)
#
#     canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
#     if target is not None:
#         cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
#         cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
#     else:
#         cv2.circle(canvas, tuple(place_seq[-1]), 2, col, -1)
#     for i in range(len(place_seq) - 1):
#         cv2.line(canvas, tuple(place_seq[i]), tuple(place_seq[i+1]), col, 1)
#
#     plt.imshow(np.swapaxes(canvas, 0, 1), interpolation='nearest', cmap=cmap, aspect='auto', vmin=canvas.min(), vmax=canvas.max())
#     return canvas


def draw_two_path(place_len, place_gt, place_pd):
    place_gt = np.round(place_gt).astype(int)
    place_pd = np.round(place_pd).astype(int)
    plt.plot(place_gt[:, 0], place_len - place_gt[:, 1], c='k', lw=2.5, label='Real Path')
    plt.plot(place_pd[:, 0], place_len - place_pd[:, 1], 'o:', c='r', lw=2.5, ms=5, label='Predicted Path')
    plt.xlim((0, place_len))
    plt.ylim((0, place_len))
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right', fontsize=14)
    ax = plt.gca()
    ax.set_aspect(1)


def draw_path_integral(place_len, place_seq, col=(255, 0, 0)):
    place_seq = np.round(place_seq).astype(int)
    cmap = cm.get_cmap('rainbow', 1000)

    canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
    if target is not None:
        cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
    else:
        cv2.circle(canvas, tuple(place_seq[-1]), 2, col, -1)
    for i in range(len(place_seq) - 1):
        cv2.line(canvas, tuple(place_seq[i]), tuple(place_seq[i+1]), col, 1)

    plt.imshow(np.swapaxes(canvas, 0, 1), interpolation='nearest', cmap=cmap, aspect='auto')
    return canvas


def draw_path_to_target_gif(file_name, place_len, place_seq, target, col=(255, 0, 0)):
    cmap = cm.get_cmap('rainbow', 1000)
    canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
    cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
    cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)

    canvas_list = []
    canvas_list.append(canvas)
    for i in range(1, len(place_seq)):
        canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
        cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
        for j in range(i):
            cv2.line(canvas, tuple(place_seq[j]), tuple(place_seq[j+1]), col, 1)
        canvas_list.append(canvas)

    imageio.mimsave(file_name, canvas_list, 'GIF', duration=0.3)


def mu_to_map_old(mu, num_interval, max=1.0):
    if len(mu.shape) == 1:
        map = np.zeros([num_interval, num_interval], dtype=np.float32)
        map[mu[0], mu[1]] = max
    elif len(mu.shape) == 2:
        map = np.zeros([mu.shape[0], num_interval, num_interval], dtype=np.float32)
        for i in range(len(mu)):
            map[i, mu[i, 0], mu[i, 1]] = 1.0

    return map


def mu_to_map(mu, num_interval):
    mu = mu / float(num_interval)
    if len(mu.shape) == 1:
        discretized_x = np.expand_dims(np.linspace(0, 1, num=num_interval), axis=1)
        max_pdf = pow(norm.pdf(0, loc=0, scale=0.02), 2)
        vec_x_before = norm.pdf(discretized_x, loc=mu[0], scale=0.02)
        vec_y_before = norm.pdf(discretized_x, loc=mu[1], scale=0.02)
        map = np.dot(vec_x_before, vec_y_before.T) / max_pdf
    elif len(mu.shape) == 2:
        map_list = []
        max_pdf = pow(norm.pdf(0, loc=0, scale=0.005), 2)
        for i in range(len(mu)):
            discretized_x = np.expand_dims(np.linspace(0, 1, num=num_interval), axis=1)
            vec_x_before = norm.pdf(discretized_x, loc=mu[i, 0], scale=0.005)
            vec_y_before = norm.pdf(discretized_x, loc=mu[i, 1], scale=0.005)
            map = np.dot(vec_x_before, vec_y_before.T) / max_pdf
            map_list.append(map)
        map = np.stack(map_list, axis=0)

    return map


def generate_vel_list(max_vel, min_vel=0.0, interval=1.0):
    vel_list = []
    max_vel_int = int(np.ceil(max_vel) + 1)
    for i in np.arange(0, max_vel_int, interval):
        for j in np.arange(0, max_vel_int, interval):
            if (np.sqrt(i ** 2 + j ** 2) <= max_vel) and (np.sqrt(i ** 2 + j ** 2) > min_vel):
                vel_list.append(np.array([i, j]))
                if i > 0:
                    vel_list.append(np.array([-i, j]))
                if j > 0:
                    vel_list.append(np.array([i, -j]))
                if i > 0 and j > 0:
                    vel_list.append(np.array([-i, -j]))
    vel_list = np.stack(vel_list)
    vel_list = vel_list.astype(np.float32)

    return vel_list
