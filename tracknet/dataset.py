from torch.utils.data import Dataset
import h5py
import os

class TrackNet(Dataset):
    def __init__(self, compiled_dataset_path, instances_per_file = 50):
        self.path = compiled_dataset_path
        self.files = os.listdir(self.path)
        self.total_len = len(self.files) * instances_per_file

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_idx = int((idx / self.total_len) * len(self.files))
        file_name = f'{file_idx:03d}.hdf5'

        instance = None
        label = None
        with h5py.File(os.path.join(self.path, file_name), 'r') as file:
            in_idx = idx % 50
            instance = file['instances'][in_idx]
            label = file['labels'][in_idx]

        return (instance, label)