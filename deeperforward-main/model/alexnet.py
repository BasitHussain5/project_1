import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cwconv import CWConv


class Alexnet(nn.Module):
    def __init__(self, in_channels=3, num_class=10, dropout=0, bias=False):
        super(Alexnet, self).__init__()
        self.num_class = num_class
        self.start_layer = 1
        self.end_layer = 4

        self.layers = nn.ModuleList([
            CWConv(in_channels, 90, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
                        CWConv(90, 200, kernel_size=5, stride=1, padding=2, bias=bias,
                               num_class=num_class, dropout=dropout)),
            nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
                        CWConv(200, 400, kernel_size=3, stride=1, padding=1, bias=bias,
                                 num_class=num_class, dropout=dropout)),
            CWConv(400, 250, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(250, 250, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
        ])

        self.optimizers = [torch.optim.AdamW(layer.parameters(), lr=0.03) for layer in self.layers]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.001)
                           for optimizer in self.optimizers]

    def forward(self, x):
        devices = [next(layer.parameters()).device for layer in self.layers]

        g_cls = None
        x = F.layer_norm(x, x.shape[1:])
        for i, layer in enumerate(self.layers):
            if i > self.end_layer:
                break
            x = x.to(devices[i])
            x, g = layer(x)
            if self.start_layer < i <= self.end_layer:
                g_cls += g.to(devices[0])
            elif i == self.start_layer:
                g_cls = g
        return g_cls

    def update(self, dataloader):
        devices = [next(layer.parameters()).device for layer in self.layers]

        criterion = nn.CrossEntropyLoss()
        self.train()
        for x, labels in dataloader:
            x, labels = x.to(devices[0]), labels.to(devices[0])
            x = F.layer_norm(x, x.shape[1:])
            for i, layers in enumerate(self.layers):
                x, labels = x.to(devices[i]), labels.to(devices[i])

                self.optimizers[i].zero_grad()
                x, g = layers(x)
                loss = criterion(g, labels)
                loss.backward()
                self.optimizers[i].step()
        for scheduler in self.schedulers:
            scheduler.step()

    def pruning(self, dataloader):
        devices = [next(layer.parameters()).device for layer in self.layers]
        corrects = [[0 for _ in range(len(self.layers))] for _ in range(len(self.layers))]
        self.eval()
        total = 0
        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                gs_cls = [0 for _ in range(len(self.layers))]
                for i, layers in enumerate(self.layers):
                    x, labels = x.to(devices[i]), labels.to(devices[i])
                    x, g = layers(x)
                    for j in range(i+1):
                        gs_cls[j] = gs_cls[j] + g.to(devices[0])
                        pred = torch.argmax(gs_cls[j], dim=1)
                        corrects[j][i] += torch.eq(pred, labels).sum().float().item()

        best_pred = 0
        best_start = 0
        best_end = 0
        for j in range(len(self.layers)):
            for i in range(0, j+1):
                if corrects[i][j] > best_pred:
                    best_pred = corrects[i][j]
                    best_start = i
                    best_end = j
        self.start_layer = best_start
        self.end_layer = best_end

        total_layer_acc = corrects[1][-1] / total
        return 100 * best_pred / total, 100 * total_layer_acc




