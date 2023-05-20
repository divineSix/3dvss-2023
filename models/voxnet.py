import os, glob

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
