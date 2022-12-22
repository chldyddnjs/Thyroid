import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png","npy",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def collate_fn(batch):
    return tuple(zip(*batch))

class BaseAugmentation:
    def __call__(self, image, mask):
        raise NotImplementedError

class Resize(BaseAugmentation):
    def __init__(self):
        super().__init__()
    
    def __call__(self,image,mask):
        # x,y,z = image.shape
        # image = image.reshape((x,y,z))
        # mask = np.argmax(mask,axis=-1)
        
        return image,mask
    
class BaseDataset(Dataset):
    def __init__(self,paths,phase='Train',transform=None):
        super().__init__()
        self.paths = pd.read_csv(paths)
        self.transform=transform
        
        self.infos = list(self.paths['patient_id'])
        self.data_paths =  list(self.paths['data'])
        self.mask_paths =  list(self.paths['mask'])
        self.phase = phase
        
    def __getitem__(self,idx):
        
        images = np.load(self.data_paths[idx])
        images = np.expand_dims(images,0)

        masks = np.load(self.mask_paths[idx])
        masks = np.expand_dims(masks,0) # cross_entropy에서 
        infos = self.infos[idx]
        if self.transform:
            transformed = self.transform(image=images,mask=masks)
            images,masks = transformed
            
        if self.phase in ['Train','Val','Test']: 
            return torch.Tensor(images).float(),torch.Tensor(masks),infos
        
    def __len__(self):
        return len(self.paths)
    