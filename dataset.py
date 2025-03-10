import pickle
import numpy as np
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, data_files, transform=None):
        self.data, self.labels = self._load_data(data_files)
        self.transform = transform
    
    def _load_data(self, files):
        data, labels = [], []
        for file in files:
            with open(file, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                data.append(batch[b'data'])
                labels.extend(batch[b'labels'])
        data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.uint8)
        print(f"Loaded data shape: {data.shape}")
        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        # CIFAR-10 is stored as (C, H, W) but needs to be (H, W, C) for transforms
        img = np.transpose(img, (1, 2, 0))
        if self.transform:
            img = self.transform(img)
        return img, label

class TestCIFAR10Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            if b'data' in data_dict:
                self.data = data_dict[b'data']
            else:
                raise ValueError("Key 'data' not found in test pickle file. Check file structure.")
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, idx