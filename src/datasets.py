import os
import gc
import numpy as np
import imageio as nd

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision import transforms

 

CIFAR10_MEAN = (0.49147478, 0.48220044, 0.4466697)
CIFAR10_STD = (0.24713175, 0.24367353, 0.26168618)

#######
from typing import Any, Callable, Optional, Tuple
import requests
from io import BytesIO
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url
from torchvision.datasets.vision import VisionDataset
class PAMAP2(VisionDataset):
    """`PAMAP2 <https://github.com/microsoft/PersonalizedFL>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``pamap2-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "pamap"
    url = "https://wjdcloud.blob.core.windows.net/dataset/cycfed/pamap.tar.gz"
    filename = "pamap.tar.gz"
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            if not os.path.exists(self.root + "/" + self.base_folder):
                self.download()

        file_path = os.path.join(self.root, self.base_folder, 'x.npy')
        self.data_all = np.load(file_path).astype(np.float32)        
        file_path = os.path.join(self.root, self.base_folder, 'y.npy')
        self.targets_all = np.load(file_path)

        indices = np.random.RandomState(seed=0).permutation(len(self.data_all))            
        self.data_all = self.data_all[indices]
        self.targets_all = self.targets_all[indices]       
        
        split_ind =int(len(self.data_all)*0.8)
        self.data = self.data_all[:split_ind]
        self.targets = self.targets_all[:split_ind]
        d_mean = self.data.mean((0,1))
        d_var = self.data.std((0,1))        
        if self.train:
            self.data = (self.data - d_mean) / d_var                                                                   
        else:
            self.data = self.data_all[split_ind:]
            self.targets = self.targets_all[split_ind:]
            self.data = (self.data - d_mean) / d_var                                                                   
        self.data = np.expand_dims(self.data, 3)
        self.data = np.einsum("itsc->istc", self.data)         
        print(self.data.shape)             
        print(self.targets.shape)             
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        download_and_extract_archive(self.url, self.root, filename=self.filename)



def get_dataset(args, verbose=1):
    
    (train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels) = eval_datasets(args)    
    
    if args.split == 0 and args.dataset != "pamap":
        train_data = np.append(train_data, valid_data, axis=0)
        train_labels = np.append(train_labels, valid_labels, axis=0)

    if verbose:    
        print("Training Data Dimensions: ", train_data.shape)
        print("[class & samples]:")    
        _unique, _counts = np.unique(np.array(train_labels).astype(int), return_counts=True)        
        print(np.asarray((_unique, _counts)).T)     

        if args.split == 1:
            print("Validation Data Dimensions: ", valid_data.shape)
            print("[class & samples]:")    
            _unique, _counts = np.unique(np.array(valid_labels).astype(int), return_counts=True)        
            print(np.asarray((_unique, _counts)).T)     

        print("Test Data Dimensions: ", test_data.shape)    
        print("[class & samples]:")    
        _unique, _counts = np.unique(np.array(test_labels).astype(int), return_counts=True)        
        print(np.asarray((_unique, _counts)).T)     

    if args.split == 0:
        return train_data, train_labels, test_data,  test_labels   
    return train_data, train_labels, valid_data, valid_labels, test_data,  test_labels   



def eval_datasets(args):
    """
    Datasets
    """ 
        
    if args.dataset == "pamap":        
        """Downloads PAMAP2 dataset and generates a unified training set (it will
        be partitioned later using the LDA partitioning mechanism."""
        save_dir = args.root_dir+"/data/pamap"
        # download dataset and load train set
        transform=transforms.Compose([transforms.ToTensor()])               
        dataset_train = PAMAP2(root=save_dir, train=True, transform=transform, download=True)
        dataset_test = PAMAP2(root=save_dir, train=False, transform=transform)    
        data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
        train_images, train_labels = next(iter(data_loader))
        data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
        test_images, test_labels = next(iter(data_loader))
        print(len(train_images), len(train_labels))
        print(len(test_images), len(test_labels))  
        dataset_train = (train_images, train_labels)        
        dataset_test = (test_images, test_labels)
        return dataset_train, (None, None), dataset_test


    elif args.dataset == "cifar10":
        save_dir = args.root_dir+"/data/temp/CIFAR10_npy_"+str(args.image_size)+"/"
        if args.data_preprocessing == 1:
            print("preprocess and save data.... ")
            root = args.root_dir+"/data"
            transform=transforms.Compose([transforms.ToTensor()])               
            dataset_train = CIFAR10(root=root, train=True, transform=transform, download=True) 
            dataset_test  = CIFAR10(root=root, train=False, transform=transform, download=True)
            data_loader = DataLoader(dataset_train, batch_size=len(dataset_train))    
            train_images, train_labels = next(iter(data_loader))
            data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))    
            test_images, test_labels = next(iter(data_loader))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            dataset_npz = np.savez_compressed(save_dir+str(args.dataset),
                                              train_images=train_images, train_labels=train_labels,
                                              test_images=test_images, test_labels=test_labels)
            print(len(train_images), len(train_labels))
            print(len(test_images), len(test_labels))                                              
            print("preprocessing is completed!")       
        ## Loading        
        dataset_npz = np.load(save_dir+str(args.dataset)+".npz")
        train_valid_images = dataset_npz['train_images']
        train_valid_labels = (dataset_npz['train_labels']).astype(int)
        indices = np.random.RandomState(seed=args.rand_seed).permutation(len(train_valid_images))
        train_valid_images = train_valid_images[indices]
        train_valid_labels = train_valid_labels[indices]
        train_images = train_valid_images[:45000]
        train_labels = train_valid_labels[:45000]
        valid_images = train_valid_images[45000:] 
        valid_labels = train_valid_labels[45000:]
        test_images  = dataset_npz['test_images']  
        test_labels  = (dataset_npz['test_labels']).astype(int)
        del dataset_npz  
        gc.collect()  
        dataset_train = (train_images, train_labels)
        dataset_valid = (valid_images, valid_labels)
        dataset_test = (test_images, test_labels)
        return dataset_train, dataset_valid, dataset_test