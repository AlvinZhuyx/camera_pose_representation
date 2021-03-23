"""
tfrecord-converter

Takes a directory of tf-records with Shepard-Metzler data
and converts it into a number of gzipped PyTorch records
with a fixed batch size.

Thanks to l3robot and versatran01 for providing initial
scripts.
"""
import os, gzip, torch
import tensorflow as tf, numpy as np, multiprocessing as mp
from functools import partial
from itertools import islice, chain
from argparse import ArgumentParser

# disable logging and gpu
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

POSE_DIM, IMG_DIM, SEQ_DIM = 5, 64, 10

def chunk(iterable, size=10):
    """
    Chunks an iterator into subsets of
    a given size.
    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def process(record):
    """
    Processes a tf-record into a numpy (image, pose) tuple.
    """
    kwargs = dict(dtype=tf.uint8, back_prop=False)
    for data in tf.python_io.tf_record_iterator(record):
        instance = tf.parse_single_example(data, {
            'frames': tf.FixedLenFeature(shape=SEQ_DIM, dtype=tf.string),
            'cameras': tf.FixedLenFeature(shape=SEQ_DIM * POSE_DIM, dtype=tf.float32)
        })

        # Get data
        images = tf.concat(instance['frames'], axis=0)
        poses  = instance['cameras']

        # Convert
        images = tf.map_fn(tf.image.decode_jpeg, tf.reshape(images, [-1]), **kwargs)
        images = tf.reshape(images, (-1, SEQ_DIM, IMG_DIM, IMG_DIM, 3))
        poses  = tf.reshape(poses,  (-1, SEQ_DIM, POSE_DIM))


        # Numpy conversion
        images, poses = images.numpy(), poses.numpy()
        yield np.squeeze(images), np.squeeze(poses)

def convert(record, batch_size, to_path):
    """
    Processes and saves a tf-record.
    """

    _, filename = os.path.split(record)
    basename, *_ = os.path.splitext(filename)
    print(basename)
    batch_process = lambda r: chunk(process(r), batch_size)

    for i, batch in enumerate(batch_process(record)):

        p = os.path.join(to_path, "{0:}-{1:02}.pt.gz".format(basename, i))
        with gzip.open(p, 'wb') as f:
            torch.save(list(batch), f)


if __name__ == '__main__':
    tf.enable_eager_execution()
    parser = ArgumentParser(description='Convert gqn tfrecords to pt files.')
    parser.add_argument('--base_dir', type=str, default="../dataset/gqn_room",
                        help='base directory of gqn dataset')
    parser.add_argument('--dataset', type=str, default="rooms_free_camera_no_object_rotations",
                        help='datasets to convert, eg. shepard_metzler_5_parts')
    parser.add_argument('--to_path', type=str, default='torch_version')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='number of sequences in each output file')
    parser.add_argument('--mode', type=str, default='test',
                        help='whether to convert train or test')
    args = parser.parse_args()

    # Find path
    base_dir = os.path.expanduser(args.base_dir)
    data_dir = os.path.join(base_dir, args.dataset, args.mode)
    to_dir = '../dataset/gqn_room/torch'
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    # Find all records
    records = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
    records = [f for f in records if "tfrecord" in f]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        f = partial(convert, batch_size=args.batch_size, to_path=to_dir)
        pool.map(f, records)
