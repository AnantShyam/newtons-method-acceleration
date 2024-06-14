# cifar10_loader.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        if train:
            for i in range(1, 6):  # Load all training batches
                data_dict = self.unpickle(os.path.join(root_dir, f'data_batch_{i}'))
                if i == 1:
                    self.data = data_dict[b'data']
                    self.labels = data_dict[b'labels']
                else:
                    self.data = np.vstack((self.data, data_dict[b'data']))
                    self.labels = np.hstack((self.labels, data_dict[b'labels']))
        else:
            data_dict = self.unpickle(os.path.join(root_dir, 'test_batch'))
            self.data = data_dict[b'data']
            self.labels = data_dict[b'labels']

    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, idx):
        image = self.data[idx].reshape(3, 32, 32).transpose((1, 2, 0))  # convert to HWC format
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
