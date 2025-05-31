import torch
from torch import nn
import ipywidgets
import matplotlib.pyplot as plt

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def accuracy(y_hat, y):
    if len(y_hat) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
    
def evaluate_accuracy_gpu(net, data_iter, device=None):
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    metrics = Accumulator(2)
    for X,y in data_iter:
        if isinstance(X, list):
            X = [i.to(device) for i in metrics]
        else:
            X = X.to(device)
        y = y.to(device)
        metrics.add(accuracy(net(X),y),y.numel())
        return metrics[0] / metrics[1]
    
def train_gpu(net, loss, updater, train_iter, test_iter, num_epoch, device, display_function):
    progress = ipywidgets.IntProgress(min=0,max=num_epoch)
    progress.value=0
    display_function(progress)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    train_metrics = []

    for epoch in range(num_epoch):
        net.train()
        metric = Accumulator(3)
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)

            out = net(X)
            net.zero_grad()
            l = loss(out, y)
            l.backward()
            updater.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(out, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

        test_acc = evaluate_accuracy_gpu(net, test_iter)

        print(f"Epoch: {epoch}, train_loss: {train_l}, train_accuracy: {train_acc}, test_accuracy: {test_acc}")
        train_metrics.append((epoch,train_l,train_acc,test_acc))
        progress.value = epoch + 1
    return train_metrics

def plot_metrics(train_metrics,y_min,y_max):
    fig, ax = plt.subplots()
    ax.plot(range(len(train_metrics)),[i[1] for i in train_metrics])
    ax.plot(range(len(train_metrics)),[i[2] for i in train_metrics])
    ax.plot(range(len(train_metrics)),[i[3] for i in train_metrics])
    ax.set_xlim(0,len(train_metrics)-1)
    ax.set_ylim(y_min,y_max)
