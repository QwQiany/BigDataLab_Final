import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp
import utils.hypergraph_2_utils as hgut
from models import HGNN
from config import get_config
#from datasets import load_feature_construct_H
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#from utils import hypergraph_utils as hgut
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
# Use default CUDA visibility so PyTorch can pick up available GPU(s).
def load_st_construct_H(
                             m_prob=1,
                             K_neigs=[1],
                             is_probH=True,
                             split_diff_scale=False,
                             use_st_feature=True,
                             use_st_feature_for_structure=True,
                            ):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """
    # init feature

    if True:
        
        # 读取新的幸福感数据集
        student_feature = pd.read_csv("processed_happiness_data.csv")
        
        print(student_feature)
        # 标签在第 1 列 (index 1)
        lbls = student_feature.iloc[:, 1].values
        # 特征从第 2 列开始 (index 2:)
        fts = student_feature.iloc[:, 2:].astype(np.float32).values

        # 转成 numpy (float32) + int label
        lbls = lbls.astype(int)

        # 划分训练集和测试集
        fts_, x_val, lbls_, y_val = train_test_split(fts, lbls, train_size=0.8, random_state=0)
        
        x = pd.DataFrame(fts_)
        y = pd.Series(lbls_)
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=0)
        idx_train = x_train.index
        idx_test = x_val.index
        
    # construct feature matrix
    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    if use_st_feature_for_structure:
        # 使用单视图构建超图
        tmp = hgut.construct_H_with_KNN(fts_, lbls_, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        print(tmp.shape)
        H = hgut.hyperedge_concat(H, tmp)
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')

    return fts_,  lbls_, idx_train, idx_test, H



#cfg = get_config('./config/config.yaml')
# initialize data
# data_dir = cfg['modelnet40_ft'] if cfg['on_dataset'] == 'ModelNet40' \
#     else cfg['ntu2012_ft']
# fts, lbls, idx_train, idx_test, H = \
#     load_feature_construct_H(data_dir,
#                              m_prob=cfg['m_prob'],
#                              K_neigs=cfg['K_neigs'],
#                              is_probH=cfg['is_probH'],
#                              split_diff_scale=False,
#                              use_mvcnn_feature=cfg['use_mvcnn_feature'],
#                              use_gvcnn_feature=cfg['use_gvcnn_feature'],
#                              #use_st_feature=True,
#                              use_mvcnn_feature_for_structure=cfg['use_mvcnn_feature_for_structure'],
#                              use_gvcnn_feature_for_structure=cfg['use_gvcnn_feature_for_structure'],
#                              #use_st_feature_for_structure=cfg['use_st_feature_for_structure'],
#                             )

fts, lbls, idx_train, idx_test, H = \
    load_st_construct_H(
                             m_prob=1,
                             K_neigs=[2],
                             is_probH=True,
                             split_diff_scale=False,
                             use_st_feature=False,
                             use_st_feature_for_structure=True,
                            )
print(H)
G = hgut.generate_G_from_H(H)
print(G)
print(G.shape)
n_class = int(lbls.max()) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            #print(preds)
            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)
            accuracy = accuracy_score(lbls[idx].cpu().numpy(), preds[idx].cpu().numpy())
            classification = classification_report(lbls[idx].cpu().numpy(), preds[idx].cpu().numpy(),digits=4)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}' )
                print(accuracy)
                print(classification)
                print(fts.shape)

            # deep copy the model
            if phase == 'val' and accuracy > best_acc:
                best_acc = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _main():


    model_ft = HGNN(in_ch=fts.shape[1],
                    n_class=n_class,
                    n_hid=128,
                    dropout=0.5)
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=0.001,
                           weight_decay=0.0005)

    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100],
                                               gamma=0.9)
    
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model_ft, criterion, optimizer, schedular, 2000, print_freq=50)

    print("\nFinal Evaluation on Test Set (Best Model):")
    model_ft.eval()
    with torch.no_grad():
        outputs = model_ft(fts, G)
        _, preds = torch.max(outputs, 1)
        # idx_test is global
        accuracy = accuracy_score(lbls[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        classification = classification_report(lbls[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy(), digits=4)
        print(f"Accuracy: {accuracy}")
        print(classification)


if __name__ == '__main__':
    _main()
