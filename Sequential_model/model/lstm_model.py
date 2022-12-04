import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append('./Sequential_Model')

from preprocess.dataloader import create_test_loader, create_train_val_loader


num_trials = 3
num_features = 13
num_outs = 2
num_epochs = 100
opt_func = torch.optim.Adam
lr = .001

# checkpoint_dir = r'./checkpoint'

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Chairrise_Classification_Base(nn.Module):
    def training_step(self, batch):
        data, labels = batch
        out = self(data)
        labels = labels.type(torch.LongTensor).to('cuda')
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        data, labels = batch
        out = self(data)
        labels = labels.type(torch.LongTensor).to('cuda')
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return dict(val_loss = loss.detach(), val_acc = acc)

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        return dict(val_loss = epoch_loss.item(), val_acc = epoch_acc.item())

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch + 1}], train_loss : {result['train_loss'] : .4f}, val_loss : {result['val_loss'] : .4f}, val_acc : {result['val_acc'] : .4f}")
    

class LSTM(Chairrise_Classification_Base):
    def __init__(self, hidden_size = 32, num_layers = 2):
        super(LSTM, self).__init__()
        self.input_size = 3
        self.out_size = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = .2
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            
            batch_first=True)

        self.out_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.out_size)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        batch_size, time_steps, seq_len = x.size()
        # print('input shape: {}'.format(x.shape))

        lstm_in = x.reshape(batch_size, -1, time_steps)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        lstm_out, _ = self.lstm(lstm_in, (h0, c0))
        # print('lstm_out size: {}'.format(lstm_out.size()))

        fc_out = self.out_network(lstm_out[:, -1, :])
        return fc_out


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):

        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

with torch.no_grad():
    def evaluate(model, val_loader):
        
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        
        return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
                
            lrs.append(get_lr(optimizer))
            sched.step()
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    return history


def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history

if __name__ == '__main__':
    model = LSTM()
    # print(model)
    # model.to(device = 'cuda')
    # sample_input = torch.randn((32, 3, 3)).to(device = 'cuda')

    # out = model(sample_input)
    # print(out.size())
    
    device = get_default_device()
    to_device(model, device)

    train_dl, val_dl = create_train_val_loader()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    print(evaluate(model, val_dl))


    






    