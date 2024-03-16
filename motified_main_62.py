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

###########################################################################################################



def train(epoch, model, dataloaders, optimizer):
    # lambd = 2 / (1 + np.exp(-10 * (epoch) / args.nepoch)) - 1
    lambd = -4 / (np.sqrt(epoch / (args.nepoch - epoch + 1)) + 1) + 4
    model.train()
    source_loader, target_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    for i in range(0, num_iter):
        source_data, source_label = next(iter_source)
        target_data, _ = next(iter_target)
        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data = target_data.cuda()
        optimizer.zero_grad()
        logit_src_1, logit_src_2, logit_tar_1, logit_tar_2 = model(source_data.float(), target_data.float())
        clc_loss_step = criterion(logit_src_2, source_label)
        rbf_djpmmd_loss_1 = MMD_loss().cmmd(logit_src_1, source_label, logit_tar_1, logit_tar_2)
        rbf_djpmmd_loss_2 = MMD_loss().marginal(logit_src_1, logit_tar_1)
        # rbf_djpmmd_loss = LMMD_loss(class_num=args.num_classes).get_loss(logit_src_1, logit_tar_1, source_label,
        #                                                                  F.softmax(logit_tar_2, dim=1))
        rbf_djpmmd_loss = args.wmmd * rbf_djpmmd_loss_2 + args.wcmmd * rbf_djpmmd_loss_1
        loss_step = clc_loss_step + lambd * args.weight * rbf_djpmmd_loss
        loss_step.backward()
        optimizer.step()
        metric_accuracy_1.update(logit_src_2.max(1)[1], source_label)
        metric_mean_1.update(loss_step)
        metric_mean_2.update(criterion(logit_src_2, source_label))
        metric_mean_5.update(clc_loss_step)
        metric_mean_6.update(rbf_djpmmd_loss)
    train_acc = metric_accuracy_1.compute()
    train_all_loss = metric_mean_1.compute()  # loss_step
    train_loss = metric_mean_2.compute()
    source_cla_loss = metric_mean_5.compute()
    djp_loss = metric_mean_6.compute()
    metric_accuracy_1.reset()
    metric_mean_1.reset()
    metric_mean_2.reset()
    metric_mean_5.reset()
    metric_mean_6.reset()
    return train_acc, train_all_loss, train_loss, source_cla_loss, djp_loss


def test(model, target_loader):
    model.eval()
    with torch.no_grad():
        iter_target = iter(dataloaders[-1])
        num_iter = len(target_loader)
        for i in range(0, num_iter):
            target_data, target_label = next(iter_target)
            target_data, target_label = target_data.cuda(), target_label.cuda()
            output2 = model.predict(target_data.float())
            metric_accuracy_2.update(output2.max(1)[1], target_label)
            metric_mean_3.update(criterion(output2, target_label))
        test_acc = metric_accuracy_2.compute()
        test_loss = metric_mean_3.compute()
        metric_accuracy_2.reset()
        metric_mean_3.reset()
        return test_acc, test_loss



if __name__ == '__main__':
    args = parse_args()
    config = vars(args)
    config_file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".yaml"
    with open(os.path.join("H:/WDAN/logs", config_file_name), "w") as file:
        file.write(yaml.dump(config))
    set_logger1()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_accuracy_1 = Accuracy(task='multiclass', num_classes=4).cuda()
    metric_accuracy_2 = Accuracy(task='multiclass', num_classes=4).cuda()
    metric_mean_1 = MeanMetric().cuda()
    metric_mean_2 = MeanMetric().cuda()
    metric_mean_3 = MeanMetric().cuda()
    metric_mean_4 = MeanMetric().cuda()
    metric_mean_5 = MeanMetric().cuda()
    metric_mean_6 = MeanMetric().cuda()
    t_test_acc = 0.0
    stop = 0
    Train_source, Train_target = load_data()
    g = torch.Generator()
    source_loader = da.DataLoader(dataset=Train_source, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    target_loader = da.DataLoader(dataset=Train_target, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    target_loader_test = da.DataLoader(dataset=Train_target, batch_size=args.batch_size, shuffle=False, generator=g)
    dataloaders = [source_loader, target_loader, target_loader_test]
    model = TICNN(num_class=args.num_classes, init_weights=args.init_weight).to(device)
    logging.info('Parameters:{}'.format(sum(param.numel() for param in model.parameters())))
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gramma)
    torch.cuda.synchronize()
    starttimem = time.time()
    losses = []
    acces = []
    test_losses = []
    test_acces = []
    for epoch in range(0, args.nepoch):
        stop += 1
        train_acc, train_all_loss, train_loss, source_cla_loss, djp_loss = train(epoch, model, dataloaders, optimizer)
        test_acc, test_loss = test(model, dataloaders)
        losses.append(train_loss.detach().cpu().numpy())
        acces.append(train_acc.detach().cpu().numpy())
        test_losses.append(test_loss.detach().cpu().numpy())
        test_acces.append(test_acc.detach().cpu().numpy())
        if t_test_acc < test_acc:
            t_test_acc = test_acc
            stop = 0
            torch.save(model.state_dict(), args.src_data+'_'+args.tar_data+'.pt')
        logging.info('{}-{}: Epoch{}, train_loss is {:.5f},test_loss is {:.5f}, train_accuracy is {:.5f},test_accuracy is {:.5f},train_all_loss is {:.5f},source_cla_loss is {:.5f},cda_loss is {:.5f}'.format(
            args.src_data, args.tar_data,
            epoch + 1, train_loss, test_loss, train_acc, test_acc, train_all_loss, source_cla_loss,
                djp_loss))
        scheduler.step()
        if stop >= args.early_stop:
            logging.info(
                '\033[1;30m Final test acc: {:.2f}% \033[0m'.format(100. * t_test_acc))
            break
    torch.cuda.synchronize()
    endtimem = time.time()
    dtime1 = endtimem - starttimem
    print("The running time: %.8s s" % dtime1)
    logging.info('\033[1;31m The running time: {:.8f}s \033[0m'.format(dtime1))
    logging.info('\033[1;31m The finally accuracy: {:.2f}% \033[0m'.format(100. * t_test_acc.detach().cpu().numpy()))
    import pandas as pd
    pd.set_option('display.max_columns', None)  # 显示完整的列
    pd.set_option('display.max_rows', None)  # 显示完整的行
    dataframe = pd.DataFrame(
        {'eval_loss': test_losses, 'eval_acc': test_acces, 'acc': acces, 'loss': losses, 'time': dtime1})
    dataframe.to_csv(args.src_data+'_'+args.tar_data+'.csv', index=False, sep=',')
    logging.info('\033[1;31m The average accuracy: {:.2f}% \033[0m'.format(100. * np.mean(np.array(test_acces[-10:]))))
    logging.info('\033[1;31m The std accuracy: {:.2f}% \033[0m'.format(100. * np.std(np.array(test_acces[-10:]))))

