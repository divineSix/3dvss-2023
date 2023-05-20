import os, glob
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for ModelNet Voxels. We can use the PyTorch DataLoader class directly after this. 
class ModelNetDataset(Dataset):
    def __init__(self, root_dir: str, num_class: int, class_idx_map: dict, split: str = "train") -> None:
        """
        Args:

        """
        super().__init__()
        self.root_dir = root_dir
        self.num_class = num_class
        self.fpaths = []
        self.class_idx_map = class_idx_map
        for cls, indx in class_idx_map.items():
            for fpath in glob.glob(os.path.join(root_dir, cls, split, "*.npy")):
                self.fpaths.append(fpath)

    # Overleading __getitem__ class
    def __getitem__(self, index):
        fpath = self.fpaths[index]
        cls_name = fpath.split("/")[-3] # Path string like so: ..../<class_name>/<split>/<class_name_00001>.npy
        cls_idx = self.class_idx_map[cls_name]
        vox_data = np.load(fpath)
        # Make into row vector
        vox_data = vox_data[np.newaxis, :]
        return {"voxel": vox_data, "cls_idx": cls_idx}

    # Overloading length class. 
    def __len__(self):
        return len(self.fpaths)

# VoxNet class
class VoxNet(torch.nn.Module):
    def __init__(self, num_class=10, input_shape=(32,32,32)):
        super(VoxNet, self).__init__()
        self.num_class = num_class
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ("conv3d_1", torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)),
            ("relu1", torch.nn.ReLU()),
            ("drop1", torch.nn.Dropout(p=0.2)),
            ("conv3d_2", torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ("relu2", torch.nn.ReLU()),
            ("pool2", torch.nn.MaxPool3d(2)),
            ("drop2", torch.nn.Dropout(p=0.3))
        ]))

        # Calculating dimensionality for MLP input vector
        temp = self.feat(torch.rand((1,1) + input_shape))
        dim = 1
        for n in temp.size()[1:]:
            dim *= n
        
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("fc1", torch.nn.Linear(dim, 128)),
            ("relu1", torch.nn.ReLU()),
            ("drop3", torch.nn.Dropout(p=0.4)),
            ("fc2", torch.nn.Linear(128, self.num_class))
        ]))
    
    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x