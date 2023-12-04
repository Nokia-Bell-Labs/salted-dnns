###
# This file is used for setting up the experiment before running main.py
###
import argparse
import torch
import torchvision

def get_parser():

    parser = argparse.ArgumentParser(description="To Set up the Experiment.")
    parser.add_argument('--data_preprocessing', default=1, help="For the first time, for each dataset, you need to set this to 1")    
    ### CIFAR-10
    parser.add_argument("--dataset", default="cifar10", help="")
    parser.add_argument('--image_size', default=32, help="Size of the Input: e.g., 32x32")
    parser.add_argument("--num_classes", default=10, help="")
    parser.add_argument('--num_input_channels', default=3, help="")
    ### PAMAP2
    # parser.add_argument("--dataset", default="pamap", help="")
    # parser.add_argument("--num_classes", default=13, help="")
    # parser.add_argument('--num_input_channels', default=1, help="")
    parser.add_argument("--split", default=0, help="In this version, this should 0")
    parser.add_argument('--with_aug',   default=False, help="For training with data augmentation")
    parser.add_argument("--epochs", default=500, help="")
    parser.add_argument("--batch_size", default=100, help="")
    parser.add_argument("--optim", default="adam", help="")
    parser.add_argument("--ler_rate", default=0.001, help="")
    parser.add_argument("--gpu_id", default=0, help="")
    parser.add_argument('--device', default='cuda', help="When using GPU, set --device='cuda'")
    parser.add_argument("--reproducibility", default=False, help="")
    parser.add_argument("--rand_seed", default=7, help="")
    parser.add_argument('--root_dir',   default="salted_dnn", help="The root directory for saving data and results.")    
    ## Salted_DNN
    parser.add_argument('--salt_size',   default=10, help="In this version, this should be the same as num_classes")    
    parser.add_argument('--salt_type',   default="cnn", help="none, cnn, or fc")    
    parser.add_argument('--salt_layer',   default=3, help="-1 for none -- For CIFAR10 1, 3, 5 for cnn or 6, 7 for fc --- For PAMAP2 cnn with 2,3,4")    
    parser.add_argument('--exp_name',   default="private_exp_1", help="The name of expriment.")    
    ## Standard DNN
    # parser.add_argument('--salt_type',   default="none", help="none, cnn, or fc")    
    # parser.add_argument('--salt_layer',   default=-1, help="-1 for none or 0-8")    
    # parser.add_argument('--exp_name',   default="standard", help="The name of expriment.")    

    return parser