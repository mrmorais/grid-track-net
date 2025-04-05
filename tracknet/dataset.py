from torch.utils.data import IterableDataset, get_worker_info
import h5py
import os

class TrackNet(IterableDataset):
    def __init__(self, path, files, debug=False, instances_per_file = 50):
        self.files = files
        self.path = path
        self.debug = debug
        self.total_len = len(self.files) * instances_per_file

    def generate(self):
        for file in self.files:
            with h5py.File(os.path.join(self.path, file), 'r') as file:
                instances = file['instances']
                labels = file['labels']

                for i in range(len(instances)):
                    instance = instances[i]
                    label = labels[i]

                    yield (instance, label)
            

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            pass
        else:
            self.files = self.files[worker_info.id::worker_info.num_workers]

        return iter(self.generate())
