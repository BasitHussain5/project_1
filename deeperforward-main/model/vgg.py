import queue
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cwconv import CWConv


class Vgg(nn.Module):
    def __init__(self, in_channels=3, num_class=10, dropout=0, bias=False, learning_rate=0.08, lr_min=0.008, devices=None):
        super(Vgg, self).__init__()
        self.num_class = num_class
        # self.start_layer = 1
        # self.end_layer = 13
        self.register_buffer('start_layer', torch.tensor(1, dtype=torch.int))
        self.register_buffer('end_layer', torch.tensor(13, dtype=torch.int))

        self.layers = nn.ModuleList([
            # Block 1
            CWConv(in_channels, 70, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(70, 70, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            # CWConv(70, 70, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            # Block 2
            nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    CWConv(70, 140, kernel_size=3, stride=1, padding=1, bias=bias,num_class=num_class, dropout=dropout)),
            CWConv(140, 140, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(140, 140, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            # Block 3
            nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
                    CWConv(140, 280, kernel_size=3, stride=1, padding=1, bias=bias,num_class=num_class, dropout=dropout)),
            CWConv(280, 280, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(280, 280, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            # Block 4
            nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
                    CWConv(280, 560, kernel_size=3, stride=1, padding=1, bias=bias,num_class=num_class, dropout=dropout)),
            CWConv(560, 560, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(560, 560, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            # Block 5
            nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
                    CWConv(560, 560, kernel_size=3, stride=1, padding=1, bias=bias,num_class=num_class, dropout=dropout)),
            CWConv(560, 560, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(560, 560, kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
        ])

        if devices is not None:
            num_layer = len(self.layers)
            num_device = len(devices)
            layer_groups = [i * num_layer // num_device for i in range(num_device)]
            layer_groups.append(num_layer)

            for i in range(num_device):
                for j in range(layer_groups[i], layer_groups[i + 1]):
                    self.layers[j].to(devices[i])

        self.optimizers = [torch.optim.Adam(layer.parameters(), lr=learning_rate) for layer in self.layers]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=lr_min)
                           for optimizer in self.optimizers]

    def forward(self, x, layer_idx=-1):
        devices = [next(layer.parameters()).device for layer in self.layers]

        g_cls = None
        x = F.layer_norm(x, x.shape[1:])
        for i, layer in enumerate(self.layers):
            if i > self.end_layer and layer_idx == -1:
                break
            x = x.to(devices[i])
            x, g = layer(x)

            # return local representation
            if layer_idx == i:
                return x

            if self.start_layer < i <= self.end_layer:
                g_cls += g.to(devices[0])
            elif i == self.start_layer:
                g_cls = g.to(devices[0])
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
                x, g = layers(x)    # normal
                # y, g = layers(x)  # memory saving
                loss = criterion(g, labels)
                loss.backward()

                self.optimizers[i].step()

                # memory saving
                # before_memory = torch.cuda.memory_allocated(device=devices[0])
                # del x, g, loss
                # torch.cuda.empty_cache()
                # after_memory = torch.cuda.memory_allocated(device=devices[0])
                # print(f"Memory: {before_memory / 1024**2:.2f} -> {after_memory/ 1024**2:.2f}")
                # x = y


            # for optimizer in self.optimizers:
            #     optimizer.step()
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
                    x = x.to(devices[i])
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
        self.start_layer = torch.tensor(best_start, dtype=torch.int)
        self.end_layer = torch.tensor(best_end, dtype=torch.int)

        total_layer_acc = corrects[1][-1] / total
        return 100 * best_pred / total, 100 * total_layer_acc

    def update_pipeline(self, dataloader, queue_size=5):
        def train_func(module, optimizer, dataloader_size, in_queue, out_queue):
            device = next(module.parameters()).device
            criterion = nn.CrossEntropyLoss()

            module.train()
            for _ in range(dataloader_size):
                x, labels = in_queue.get()
                x, labels = x.to(device), labels.to(device)

                optimizer.zero_grad()
                x, g = module(x)
                if out_queue is not None:
                    out_queue.put((x, labels))

                loss = criterion(g, labels)
                loss.backward()
                optimizer.step()
        # end of train_func

        queues = [queue.Queue(queue_size) for _ in range(len(self.layers))]
        queues += [None]
        threads = [threading.Thread(target=train_func,
                                    args=(layer, optimizer, len(dataloader), queues[i], queues[i+1]))
                   for i, (layer, optimizer) in enumerate(zip(self.layers, self.optimizers))]

        for t in threads:
            t.start()

        for x, labels in dataloader:
            x = x.to(next(self.layers[0].parameters()).device)
            x = F.layer_norm(x, x.shape[1:])
            queues[0].put((x, labels))

        for t in threads:
            t.join()

        for scheduler in self.schedulers:
            scheduler.step()

    def test_local_acc(self, dataloader):
        devices = [next(layer.parameters()).device for layer in self.layers]
        corrects = [0 for _ in range(len(self.layers))]
        self.eval()
        total = 0
        with torch.no_grad():
            for x, labels in dataloader:
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                for i, layers in enumerate(self.layers):
                    x, labels = x.to(devices[i]), labels.to(devices[i])
                    x, g = layers(x)
                    pred = torch.argmax(g, dim=1)
                    corrects[i] += torch.eq(pred, labels).sum().float().item()

        return [100 * corrects[i] / total for i in range(len(self.layers))]