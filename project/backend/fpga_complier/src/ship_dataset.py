import os
from torch.utils.data import Dataset
from src.data_augmentation import *
import glob
import numpy as np

class SHIPDataset(Dataset):
    def __init__(self, root_path=" ", image_size=416, is_training=True):
        if is_training:
            contents = glob.glob(os.path.join(root_path, 'trainingset/images'+os.sep+'*jpg'))
        else:
            contents = glob.glob(os.path.join(root_path, 'testingset/images'+os.sep+'*jpg'))
        self.ids = [id.strip().split(os.sep)[-1][0:-4] for id in contents]
        self.root_path = root_path
        self.image_size = image_size
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        if self.is_training:
            image_path = os.path.join(self.root_path, "trainingset/images", "{}.jpg".format(id))
            label_path = os.path.join(self.root_path, "trainingset/labels", "{}.txt".format(id))
        else:
            image_path = os.path.join(self.root_path, "testingset/images", "{}.jpg".format(id))
            label_path = os.path.join(self.root_path, "testingset/labels", "{}.txt".format(id))

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        objects = np.loadtxt(label_path)
        if len(objects.shape) == 1:
            objects = objects[None, :]

        if self.is_training:
            transformations = Compose([VerticalFlip(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))
        return np.array(image, dtype=np.float32)[None, :, :], np.array(objects, dtype=np.float32)
