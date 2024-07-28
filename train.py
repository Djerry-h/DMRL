import argparse
import torch
import torch.optim as optim
import time
from model import AMNet
from copy import deepcopy
from config import *
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(data, model, criterion, optimizer, label, beta=.6):
    anomaly, normal = label
    idx_train = data.train_mask
    model.train()
    optimizer.zero_grad()
    output, bias_loss = model(data.x, data.edge_index, label=(data.train_mask & anomaly, data.train_mask & normal))
    loss_train = criterion(output[idx_train], data.y[idx_train].long()) + bias_loss * beta
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), output


def sample_mask(idx, l):
        """Create mask."""
        mask = torch.zeros(l)
        mask[idx] = 1
        return torch.as_tensor(mask, dtype=torch.bool)


def main(args, exp_num=0):
    if args.dataset == 'elliptic':
        data = pickle.load(open('/home/hdou/model/conference-diff-data/AMNet-main/dataset/elliptic.dat', 'rb'))
        data = data.to(device)
        
    elif args.dataset == 'yelp':
        data = pickle.load(open('/home/hdou/model/first/MC-AGCN/AMNet-main/dataset/yelp.dat', 'rb'))
        data = data.to(device)
    elif args.dataset == 'weibo':
        data = torch.load("/home/hdou/model/conference-diff-data/AMNet-main/dataset/weibo.pt").to(device)
        # data = torch.load("/home/hdou/model/conference-diff-data/AMNet-main/dataset/BlogCatalog").to(device)
        sample_number = len(data.y) 
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 计算每个子集的样本数量
        train_size = int(0.4 * sample_number)
        val_size = int(0.2 * sample_number)
        test_size = sample_number - train_size - val_size
        indices = np.arange(sample_number)
        np.random.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        train_mask = sample_mask(train_idx, sample_number)
        val_mask = sample_mask(val_idx, sample_number) 
        test_mask = sample_mask(test_idx, sample_number)
        data.train_mask = train_mask.to(device)
        data.test_mask = test_mask.to(device)
        data.val_mask = val_mask.to(device)
        
    elif args.dataset == 'Amazon':
        data = torch.load("/home/hdou/model/conference-diff-data/AMNet-main/dataset/Amazon").to(device)
    elif args.dataset == 'tfinance':
        data = torch.load("/home/hdou/model/conference-diff-data/AMNet-main/dataset/tfinance.pt").to(device)
        
        
    net = AMNet(in_channels=data.x.shape[1], hid_channels=params_config['hidden_channels'], num_class=2)
    net.to(device)
    optimizer = optim.Adam([
        dict(params=net.filters.parameters(), lr=params_config['lr_f']),
        dict(params=net.filters1.parameters(), lr=params_config['lr_f']),
        dict(params=net.lin, lr=params_config['lr'], weight_decay=params_config['weight_decay']),
        dict(params=net.attn, lr=params_config['lr'], weight_decay=params_config['weight_decay'])
    ])
    weight = (1 - data.y[data.train_mask]).sum().item() / data.y[data.train_mask].sum().item()
    weights = torch.Tensor([1., weight])
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    anomaly = (data.y == 1)
    normal = (data.y == 0)
    label = (anomaly, normal)
    
    c = 0
    auc_pr_best = 0
    auc_roc_best = 0
    auc_roc_test_epoch = 0
    auc_pr_test_epoch = 0
    best_net = None

    for epoch in range(params_config['epochs']):
        loss, output = train(data, net, criterion, optimizer, label, beta=params_config['beta'])
        auc_roc_val, auc_pr_val = net.evaluating(data.x, data.y, data.edge_index, data.val_mask)
       
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == 0:
            print('Epoch:{:04d}\tloss:{:.4f}\tVal AUC-ROC:{:.4f}\tVal AUC-PR:{:.4f}'
                  '\ttest AUC-ROC:{:.4f}\ttest AUC-PR:{:.4f}\tTrain Time:{:.4f}\tEval Time:{:.4f}'
                  .format(epoch + 1, loss, auc_roc_val, auc_pr_val, auc_roc_test_epoch, auc_pr_test_epoch, training_time, evaluation_time))
        
        if auc_pr_val >= auc_pr_best:
            auc_pr_best = auc_pr_val
            auc_roc_best = auc_roc_val
            auc_roc_test_epoch, auc_pr_test_epoch = net.evaluating(data.x, data.y, data.edge_index, data.test_mask)
            best_net = deepcopy(net)
            c = 0
        else:
            c += 1
        if c == params_config['patience']:
            break
      

    auc_roc_test_exp, auc_pr_test_exp = best_net.evaluating(data.x, data.y, data.edge_index, data.test_mask)
     
    return auc_roc_test_exp, auc_pr_test_exp, attn_values_epoch, output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp', help='Dataset [yelp, elliptic, weibo, quest, Amazon, ACM]')
    parser.add_argument('--exp_num', type=int, default=21, help='Default Experiment Number')
    parser.add_argument('--eval_interval', type=int, default=100)
    args = parser.parse_args(args=[])
    params_config = dataset_config[args.dataset]
    auc_roc_list = []
    auc_pr_list = []
    
    for i in range(args.exp_num):
        auc_roc_test, auc_pr_test, attn_values_epoch ,output= main(args, exp_num=i)
        auc_roc_list.append(auc_roc_test)
        auc_pr_list.append(auc_pr_test)

    print("AUC ROC Mean:{:.5f}\tStd:{:.5f}\tAUC PR Mean:{:.5f}\tStd:{:.5f}".format(np.mean(auc_roc_list),
                                                                                   np.std(auc_roc_list),
                                                                                   np.mean(auc_pr_list),
                                                                                   np.std(auc_pr_list)))
   