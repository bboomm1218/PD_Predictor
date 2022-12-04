import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append('./Sequ')
dataset_dir = 'C://Users/bboom/Capstone_Design/Sequential_Model/dataset/'

from preprocess.dataloader import create_test_loader
from preprocess.dataset import chairrise_dataset
from model.lstm_model import to_device, LSTM, evaluate, DeviceDataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_label(data, model):
    xb = to_device(data.unsqueeze(0), device)
    yb = model(xb)

    _, preds = torch.max(yb, dim = 1)

    return preds[0].item()


if __name__ == '__main__':
    model = to_device(LSTM(), device)
    model.load_state_dict(torch.load('C://Users/bboom/Capstone_Design/Sequential_Model/model/HC_PD_lstm.pth'))

    test_loader = create_test_loader()
    test_dl = DeviceDataLoader(test_loader, device)

    test_set = chairrise_dataset(dataset_dir, 'test')

    print(evaluate(model, test_dl))

    targets = []
    preds = []
    for i in range(len(test_set)):

        preds.append(predict_label(test_set[i][0], model))
        targets.append(int(test_set[i][1]))
        
        
    # print(targets)
    # print(preds)
    
    print(Counter(preds))

    cm = confusion_matrix(targets, preds)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    plt.figure(figsize = (10, 6))
    plt.title('LSTM')
    ax = sns.heatmap(cm, annot = True, cmap = 'binary', fmt = '.2%', cbar = False, annot_kws = {'fontsize' : 15})
    ax.set_xlabel('\nPrediction')
    ax.set_ylabel('Label')

    ax.xaxis.set_ticklabels(['CG', 'PD'])
    ax.yaxis.set_ticklabels(['CG', 'PD'])
    plt.show()
    
    print(classification_report(targets, preds))
    