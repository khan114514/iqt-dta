import torch
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset


def torch_load_compat(path):
    # PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks
    # loading processed PyG Data objects saved by this repository.
    return torch.load(path, weights_only=False)

class GNNDataset(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch_load_compat(self.processed_paths[0])
        else:
            self.data, self.slices = torch_load_compat(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass

if __name__ == "__main__":
    dataset = GNNDataset('data/davis')


class MergedGNNDataset(Dataset):
    """
    Presents legacy train/test processed files as one unified index space.
    Future split manifests can index into this merged dataset without changing
    the MGraphDTA backbone or preprocessing format.
    """

    def __init__(self, root):
        self.root = root
        self.train_dataset = GNNDataset(root, train=True)
        self.test_dataset = GNNDataset(root, train=False)
        self.train_size = len(self.train_dataset)
        self.test_size = len(self.test_dataset)
        self.total_size = self.train_size + self.test_size

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.total_size + idx
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} is out of range for merged dataset of size {self.total_size}.")

        if idx < self.train_size:
            return self.train_dataset[idx]
        return self.test_dataset[idx - self.train_size]


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]
