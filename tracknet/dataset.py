from torch.utils.data import Dataset
import h5py
import time
import os

class TrackNet(Dataset):
    def __init__(self, path, files, debug=False, instances_per_file = 50):
        self.files = files
        self.path = path
        self.debug = debug
        self.total_len = len(self.files) * instances_per_file

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if (self.debug):
            start = time.time()

        file_idx = int((idx / self.total_len) * len(self.files))
        file_name = self.files[file_idx]

        instance = None
        label = None
        with h5py.File(os.path.join(self.path, file_name), 'r') as file:
            in_idx = idx % 50
            instance = file['instances'][in_idx]
            label = file['labels'][in_idx]

        if (self.debug):
            end = time.time()
            print(f"Took {end - start} to load {in_idx}")

        return (instance, label)
