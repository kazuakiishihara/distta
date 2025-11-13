import glob
import nibabel as nib
import numpy as np
import os
import pickle
import random
from scipy import ndimage

import torch
from torch.utils.data import Dataset

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms, img_size=(192, 224, 160)):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = np.transpose(x, (1, 2, 0)), np.transpose(y, (1, 2, 0))
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms, img_size=(192, 224, 160)):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = np.transpose(x, (1, 2, 0)), np.transpose(y, (1, 2, 0))
        x_seg, y_seg = np.transpose(x_seg, (1, 2, 0)), np.transpose(y_seg, (1, 2, 0))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

class LPBABrainDatasetS2S(Dataset):
    def __init__(self, data_path, transforms, img_size=(128, 128, 128)):
        self.paths = data_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = resize_volume(x, self.img_size), resize_volume(y, self.img_size)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

class LPBABrainInferDatasetS2S(Dataset):
    def __init__(self, data_path, transforms, img_size=(128, 128, 128)):
        self.paths = data_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)

    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = resize_volume(x, self.img_size), resize_volume(y, self.img_size)
        x_seg, y_seg = resize_volume(x_seg, self.img_size, order=0), resize_volume(y_seg, self.img_size, order=0)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x) # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

class IXIir(Dataset):
    def __init__(self, data_path, transforms, img_size=(192, 224, 160), num_pairs_per_epoch=400):
        self.paths = data_path
        self.transforms = transforms
        self.img_size = img_size
        self.num_pairs = num_pairs_per_epoch

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):
        x_index, y_index = random.sample(range(len(self.paths)), 2)
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, y = np.transpose(x, (1, 2, 0)), np.transpose(y, (1, 2, 0))
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

class IXIirInfer(Dataset):
    def __init__(self, data_path, transforms, img_size=(192, 224, 160)):
        self.paths = data_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg = pkload(path_x)
        y, y_seg = pkload(path_y)
        x, x_seg = np.transpose(x, (1, 2, 0)), np.transpose(x_seg, (1, 2, 0))
        y, y_seg = np.transpose(y, (1, 2, 0)), np.transpose(y_seg, (1, 2, 0))
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

class CLMIirInfer(Dataset):
    """
    Inter-patient Registration
    """
    def __init__(self, data_path, transforms, img_size=(192, 224, 160)):
        self.paths = data_path # Tuple: (Head, Segmentation, BrainMask)
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)*(len(self.paths)-1)
    
    def __getitem__(self, index):
        x_index = index//(len(self.paths)-1)
        s = index%(len(self.paths)-1)
        y_index = s+1 if s >= x_index else s
        path_x = self.paths[x_index]
        path_y = self.paths[y_index]
        x, x_seg, x_brainmask = nib.load(path_x[0]).get_fdata(), nib.load(path_x[1]).get_fdata(), nib.load(path_x[2]).get_fdata()
        y, y_seg, y_brainmask = nib.load(path_y[0]).get_fdata(), nib.load(path_y[1]).get_fdata(), nib.load(path_y[2]).get_fdata()
        x, x_seg, x_brainmask = np.squeeze(x), np.squeeze(x_seg), np.squeeze(x_brainmask)
        y, y_seg, y_brainmask = np.squeeze(y), np.squeeze(y_seg), np.squeeze(y_brainmask)
        x = x * (x_brainmask > 0)
        y = y * (y_brainmask > 0)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x, x_seg = np.transpose(x, (2, 1, 0)), np.transpose(x_seg, (2, 1, 0))
        y, y_seg = np.transpose(y, (2, 1, 0)), np.transpose(y_seg, (2, 1, 0))
        x, x_seg = np.flip(x, axis=(0,2)), np.flip(x_seg, axis=(0,2))
        y, y_seg = np.flip(y, axis=(0,2)), np.flip(y_seg, axis=(0,2))
        x, y = resize_volume(x, self.img_size), resize_volume(y, self.img_size)
        x_seg, y_seg = resize_volume(x_seg, self.img_size, order=0), resize_volume(y_seg, self.img_size, order=0)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

class CLMIar(Dataset):
    """
    Atlas-based Registration
    """
    def __init__(self, data_path, atlas_path, transforms, img_size=(192, 224, 160)):
        self.paths = data_path # Tuple: (Head, Segmentation, BrainMask)
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_brainmask = nib.load(path[0]).get_fdata(), nib.load(path[2]).get_fdata()
        y, y_brainmask = np.squeeze(y), np.squeeze(y_brainmask)
        y = y * (y_brainmask > 0)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x, y = np.transpose(x, (1, 2, 0)), np.transpose(y, (2, 1, 0))
        y = np.flip(y, axis=(0,2))
        y = resize_volume(y, self.img_size)
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

class CLMIarInfer(Dataset):
    """
    Atlas-based Registration
    """
    def __init__(self, data_path, atlas_path, transforms, img_size=(192, 224, 160)):
        self.paths = data_path # Tuple: (Head, Segmentation, BrainMask)
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.img_size = img_size

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg, y_brainmask = nib.load(path[0]).get_fdata(), nib.load(path[1]).get_fdata(), nib.load(path[2]).get_fdata()
        y, y_seg, y_brainmask = np.squeeze(y), np.squeeze(y_seg), np.squeeze(y_brainmask)
        y = y * (y_brainmask > 0)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x, x_seg = np.transpose(x, (1, 2, 0)), np.transpose(x_seg, (1, 2, 0))
        y, y_seg = np.transpose(y, (2, 1, 0)), np.transpose(y_seg, (2, 1, 0))
        y, y_seg = np.flip(y, axis=(0,2)), np.flip(y_seg, axis=(0,2))
        y, y_seg = resize_volume(y, self.img_size), resize_volume(y_seg, self.img_size, order=0)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

def resize_volume(img, shape, order=1):
    """
    Resize a 3D volume to the desired shape using interpolation.
    
    Args:
        img (np.ndarray): The input 3D volume, assuming that shape is (depth, height, width).
        shape (tuple): The desired output shape (depth, height, width).
        
    Returns:
        np.ndarray: The resized 3D volume.
    """
    # Set the desired dimensions
    desired_depth  = shape[0] # Desired depth
    desired_height = shape[1] # Desired height
    desired_width  = shape[2] # Desired width

    # Get the current dimensions of the input volume
    current_depth  = img.shape[0] # Current depth
    current_height = img.shape[1] # Current height
    current_width  = img.shape[2] # Current width

    # Compute the scaling factors for each dimension
    depth  = current_depth  / desired_depth  # Depth scaling factor
    width  = current_width  / desired_width  # Width scaling factor
    height = current_height / desired_height # Height scaling factor

    depth_factor  = 1 / depth  # Inverse depth scaling factor
    width_factor  = 1 / width  # Inverse width scaling factor
    height_factor = 1 / height # Inverse height scaling factor

    # Resize the volume using scipy's ndimage.zoom function
    img = ndimage.zoom(img, (depth_factor, height_factor, width_factor), order=order) # Linear interpolation
    return img  # Return the resized volume
