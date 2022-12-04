
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from functools import partial


sys.path.append('../')

from lstm_model import LSTM, fit_one_cycle, get_default_device, DeviceDataLoader, to_device, evaluate, create_train_val_loader

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of Epochs')
    plt.show()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('val_loss') for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Batch No.')
    plt.show()

if __name__ == '__main__':
    num_epochs = 50
    opt_func = torch.optim.Adam
    # lr = .001
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    model = LSTM()


    device = get_default_device()

    train_dl, val_dl = create_train_val_loader()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    to_device(model, device)

# print(evaluate(model, val_dl))
    history = fit_one_cycle(num_epochs, max_lr, model, train_dl, val_dl, grad_clip = grad_clip, weight_decay = weight_decay, opt_func = opt_func)

    plot_accuracies(history)

    plot_losses(history)

    plot_lrs(history)

    torch.save(model.state_dict(), './Sequential_Model/model/HC_PD_lstm.pth')