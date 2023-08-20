# -*- coding: utf-8 -*-
"""Loading dataset
"""
# Author: Partha Pratim Banik <ppbanik006@gmail.com>
# License: BSD 2 clause

import os
import torch
import shutil
import requests
import lightning.pytorch as pl
# from pygod.utils import load_data
from torch_geometric.utils import to_dense_adj
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import NeighborLoader

# class DatasetBase(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, file_path: str = None, model=None):
#         """
#         '''
#         inj_cora dataset:
#         pgv100_data:  Data(x=[2708, 1433], edge_index=[2, 11060], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
#         ('x', tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#                 [0., 0., 0.,  ..., 0., 0., 0.],
#                 [0., 0., 0.,  ..., 0., 0., 0.],
#                 ...,
#                 [0., 0., 0.,  ..., 0., 0., 0.],
#                 [0., 0., 0.,  ..., 0., 0., 0.],
#                 [0., 0., 0.,  ..., 0., 0., 0.]]))
#         ('edge_index', tensor([[   0,    0,    0,  ...,  869,  127, 1674],
#                 [ 633, 1862, 2582,  ..., 1732,  214,  438]]))
#         ('y', tensor([0, 0, 0,  ..., 0, 0, 0]))
#         ('train_mask', tensor([ True,  True,  True,  ..., False, False, False]))
#         ('val_mask', tensor([False, False, False,  ..., False, False, False]))
#         ('test_mask', tensor([False, False, False,  ...,  True,  True,  True]))
#         '''
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         super().__init__()
#         data = torch.load(file_path)
#         # self.num_neigh = model.num_layers
#         # loader = NeighborLoader(data,
#         #                         self.num_neigh,
#         #                         batch_size=self.batch_size)
        
#         self.x = data.x
#         self.y = data.y
#         self.edge_index = data.edge_index
#         self.train_mask = data.train_mask
#         self.val_mask = data.val_mask
#         self.test_mask = data.test_mask
#         self.s = to_dense_adj(data.edge_index)[0]
#         print("shape self.s= ", self.s.shape)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         sample = {'x':self.x[idx,:],
#                   'y':self.y[idx],
#                   's':self.s[idx,:],
#                   'train_mask':self.train_mask[idx],
#                   'val_mask':self.val_mask[idx],
#                   'test_mask':self.test_mask[idx]}
#         return sample


class DataSet(pl.LightningDataModule):
    def __init__(self, name: str = "inj_cora", cache_dir: str = None, batch_size: int = 0,
                 num_layers: int = 4, num_neigh: int = -1):
        super().__init__()
        self.cache_dir = cache_dir
        self.name = name
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
        self.file_path = os.path.join(self.cache_dir, self.name+'.pt')
        self.zip_path = os.path.join(self.cache_dir, self.name+'.pt.zip')
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_neigh = num_neigh
        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')
        self.data = None
        # self.db = None

    def prepare_data(self):
        '''Downloading Dataset /
        if necessary
        '''
        if not os.path.exists(self.file_path):
            url = "https://github.com/pygod-team/data/raw/main/" + self.name + ".pt.zip"
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise RuntimeError("Failed downloading url %s" % url)
            with open(self.zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            shutil.unpack_archive(self.zip_path, self.cache_dir)
    
    def setup(self):
        '''loading dataset from local path
        '''
        data = torch.load(self.file_path)
        data.s = to_dense_adj(data.edge_index)[0] # process_graph
        self.data = data
        # self.db = DatasetBase(file_path=self.file_path)
        
    def train_dataloader(self):
        if self.batch_size == 0:
            self.batch_size = self.data.x.shape[0]
        return NeighborLoader(self.data, self.num_neigh, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)


# from torch_geometric.loader import NeighborLoader

# class TestOnNL:
#     def __init__(self, name, num_layers=4, num_neigh=-1, batch_size=0):
#         super().__init__()
#         data=load_data(name)
#         self.num_layers = num_layers
#         self.num_neigh = num_neigh
#         self.batch_size = batch_size
#         if self.batch_size == 0:
#             self.batch_size = data.x.shape[0]
#             print("self.batch_size: ", self.batch_size)
#         if type(num_neigh) is int:
#             self.num_neigh = [num_neigh] * self.num_layers
#         elif type(num_neigh) is list:
#             if len(num_neigh) != self.num_layers:
#                 raise ValueError('Number of neighbors should have the '
#                                  'same length as hidden layers dimension or'
#                                  'the number of layers.')
#             self.num_neigh = num_neigh
#         print("self.num_neigh: ", self.num_neigh)
#         #self.num_neigh depends of num_layers
#         loader = NeighborLoader(data,
#                                 self.num_neigh,
#                                 batch_size=self.batch_size)
#         print("loader: ", loader)
#         print("loader type: ", type(loader))
#         for idx, sampled_data in enumerate(loader):
#             batch_size = sampled_data.batch_size
#             node_idx = sampled_data.n_id
#             print("%d."%(idx))
#             print("batch_size: ", batch_size)
#             print("node_idx: ", node_idx)
#             print("sampled_data: ", sampled_data)
#             print("sampled_data type: ", type(sampled_data))