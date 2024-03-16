import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# from weight_init import Laplace_fast as fast
from weight_init import Shannon_fast as fast1
from weight_init1 import Harmonice_fastv2 as fast
from weight_init import Morlet_fast as fast2
import torch
import torch.nn as nn
import argparse
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy
from loss.djpmmd import MMD_loss
import time
import torch.nn.functional as F
# from lmmd import LMMD_loss
import os
from lightning_fabric.utilities.seed import seed_everything
import logging
from set_logger import set_logger1
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from timm.loss import LabelSmoothingCrossEntropy

def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--src_data', type=str, default="wx1920kdata900_data", help='')
    parser.add_argument('--src_label', type=str, default="wx1920kdata900_label", help='')
    parser.add_argument('--tar_data', type=str, default="gt300data_data", help='')
    parser.add_argument('--tar_label', type=str, default="gt300data_label", help='')
    ###########################################################################################
    parser.add_argument('--seed', type=int, help='Seed', default=3407)
    parser.add_argument('--wmmd', type=float,  help='', default=0.9)
    parser.add_argument('--wcmmd', type=float, help='', default=0.1)
    parser.add_argument('--step', type=int, help='', default=100)
    parser.add_argument('--gramma', type=float, help='', default=0.1)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of the training process')
    parser.add_argument('--nepoch', type=int, default=200, help='max number of epoch')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='initialization list')
    parser.add_argument('--early_stop', type=int, help='Early stopping number', default=200)
    parser.add_argument('--weight', type=float, help='Weight for adaptation loss', default=0.5)
    #################################################################################################
    parser.add_argument('--filter', type=str, default="L", help='')
    parser.add_argument('--eps', type=float,  help='', default=-0.1)
    parser.add_argument('--src_frequery', type=int, help='', default=20000)
    parser.add_argument('--tar_frequery', type=int, help='', default=100000)
    parser.add_argument('--mode', type=str, default="maxmin", help='')
    parser.add_argument('--init_weight', type=str2bool, nargs='?', const=True, default=True)
    ##############################################################################




    args = parser.parse_args()
    return args


class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)


def load_data():
    base_path = r'H:\WDAN'
    source_data = np.load(os.path.join(base_path, args.src_data)+'.npy')
    source_label = np.load(os.path.join(base_path, args.src_label)+'.npy').argmax(axis=-1)
    target_data = np.load(os.path.join(base_path, args.tar_data)+'.npy')
    target_label = np.load(os.path.join(base_path, args.tar_label)+'.npy').argmax(axis=-1)
    # source_data = MinMaxScaler().fit_transform(source_data.T).T    #最大最小归一化
    # target_data = MinMaxScaler().fit_transform(target_data.T).T
    source_data = StandardScaler().fit_transform(source_data.T).T    #最大最小归一化
    target_data = StandardScaler().fit_transform(target_data.T).T
    source_data = np.expand_dims(source_data, axis=1)
    target_data = np.expand_dims(target_data, axis=1)
    Train_source = Dataset(source_data, source_label)
    Train_target = Dataset(target_data, target_label)
    return Train_source, Train_target


###############################################################








###################################################################################################3

class TICNN(nn.Module):
    def __init__(self, num_class=4, init_weights=True):
        super(TICNN, self).__init__()
        self.feature_layers1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=1),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.Softshrink(lambd=0.5),
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.feature_layers11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=1),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.Softshrink(lambd=0.5),
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.feature_layers2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.Softshrink(lambd=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.feature_layers3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.Softshrink(lambd=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.feature_layers4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # 32, 24, 24
            nn.BatchNorm1d(128),
            nn.Softshrink(lambd=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.pooling_layers = nn.Sequential(
            nn.AdaptiveMaxPool1d(1)
        )
        self.n_features = 128
        self.fc1 = nn.Linear(self.n_features, 64)
        self.fc2 = nn.Linear(64, num_class)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for name, can in self.named_children():
            if name == 'feature_layers1':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):
                            if args.filter == 'L':
                                m.weight.data = fast(out_channels=16, kernel_size=64, eps=args.eps,
                                                     frequency=args.src_frequery, mode=args.mode).forward()
                            elif args.filter == 'S':
                                m.weight.data = fast1(out_channels=16, kernel_size=64, eps=args.eps).forward()
                            elif args.filter == 'M':
                                m.weight.data = fast2(out_channels=16, kernel_size=64, eps=args.eps,
                                                      mode=args.mode).forward()
                            nn.init.constant_(m.bias.data, 0.0)
            elif name == 'feature_layers11':
                for m in can.modules():
                    if isinstance(m, nn.Conv1d):
                        if m.kernel_size == (64,):
                            if args.filter == 'L':
                                m.weight.data = fast(out_channels=16, kernel_size=64, eps=args.eps,
                                                     frequency=args.src_frequery, mode=args.mode).forward()
                            elif args.filter == 'S':
                                m.weight.data = fast1(out_channels=16, kernel_size=64, eps=args.eps).forward()
                            elif args.filter == 'M':
                                m.weight.data = fast2(out_channels=16, kernel_size=64, eps=args.eps,
                                                      mode=args.mode).forward()
                            nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x, y):
        x = self.feature_layers1(x)
        x = self.feature_layers2(x)
        x = self.feature_layers3(x)
        x = self.feature_layers4(x)
        x = self.pooling_layers(x)
        x1 = self.fc1(x.squeeze())
        x2 = self.fc2(x1)
        y = self.feature_layers11(y)
        y = self.feature_layers2(y)
        y = self.feature_layers3(y)
        y = self.feature_layers4(y)
        y = self.pooling_layers(y)
        y1 = self.fc1(y.squeeze())
        y2 = self.fc2(y1)
        return x1, x2, y1, y2


    def predict(self, x):
        x = self.feature_layers11(x)
        x = self.feature_layers2(x)
        x = self.feature_layers3(x)
        x = self.feature_layers4(x)
        x = self.pooling_layers(x)
        x = self.fc1(x.squeeze())
        x = self.fc2(x)
        return x
