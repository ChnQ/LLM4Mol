import json
import torch
from torch.utils.data import Dataset as TorchDataset


# ChatGPT outputs
class MolDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, encodings):
        self.encodings = encodings
        with open(dataset_path, 'r', encoding='UTF-8') as f:
            self.all_infos = json.load(f)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        if self.all_infos[idx]['label'] == '1':
            label = 1
        else:
            label = 0
        item['labels'] = torch.tensor(label)
        item['index'] = self.all_infos[idx]['index']

        return item

    def __len__(self):
        return len(self.all_infos)


# SMILES
class MolDatasetV2(torch.utils.data.Dataset):
    def __init__(self, dataset_path, encodings):
        self.encodings = encodings
        with open(dataset_path, 'r', encoding='UTF-8') as f:
            self.all_infos = json.load(f)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        if self.all_infos[idx]['label'] == '-1':
            label = 0
        else:
            label = 1
        item['labels'] = torch.tensor(label)

        return item

    def __len__(self):
        return len(self.all_infos)