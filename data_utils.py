import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
import utils
import os
    

class CustomDataset(Dataset):
    def __init__(self, brands: list, filenames_features: list, dirpath_extracted_features, device, filename_target=utils.FILENAME_RELEVANCE_WINDOW, 
                 log_target=False, center_target=False, filter_uniform_features=False, max_datapoints=None):
        self.log_target = log_target
        self.center_target = center_target
        self.device = device
        self.max_datapoints = max_datapoints
        self.features = torch.concat([torch.concat([torch.load(os.path.join(dirpath_extracted_features, brand, filename_feature), map_location=device) 
                         for filename_feature in filenames_features], dim=1)
                            for brand in brands], dim=0)
        self.target = torch.concat([torch.load(os.path.join(dirpath_extracted_features, brand, filename_target), map_location=device) 
                            for brand in brands], dim=0)
        if self.max_datapoints:
            self.features = self.features[:self.max_datapoints]
            self.target = self.target[:self.max_datapoints]
        if log_target:
            self.target = torch.log(self.target)
        if center_target:
            self.centering_shift = self.target.mean()
            self.target = self.target - self.centering_shift
        self.filter_uniform_features = filter_uniform_features
        if self.filter_uniform_features:
            self.features = self.features[:,(self.features != self.features[0]).any(0)]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]