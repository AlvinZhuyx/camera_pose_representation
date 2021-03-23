import collections, os, io
from PIL import Image
import torch, gzip
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import random
import numpy as np

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

def transform_img(img):
    return img / 255 * 2.0 - 1.0


class RoomDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.file_list = os.listdir(self.root_dir)
        self.file_list.sort()
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        cur_file = self.file_list[idx]
        scene_path = os.path.join(self.root_dir, cur_file)
        with gzip.open(scene_path, 'rb') as f:
            data = torch.load(f)
        images = []
        viewpoints = []
        for img, v in data:
            images.append(img)
            viewpoints.append(v)
        images = np.array(images, dtype=np.float32)
        viewpoints = np.array(viewpoints, dtype=np.float32)
        if self.transform:
            images = self.transform(images)

        return images, viewpoints


def sample_batch(x_data, v_data, mode="train", seed=None):
    random.seed(seed)
    if mode == "train":
        query_idx = random.randint(6, 8)
    else:
        query_idx = 9

    # Sample view
    x, v = x_data[:, :6], v_data[:, :6]
    # Sample query view
    x_q, v_q = x_data[:, query_idx], v_data[:, query_idx]
    return x, v, x_q, v_q
