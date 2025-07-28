import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cwconv import CWConv

import threading
import queue


NO_SHORTCUT = 0
ADD_SHORTCUT = 1
CONCAT_SHORTCUT = 2


# InfoNCE loss
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize feature vectors
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate positive and negative pairs
        z = torch.cat([z_i, z_j], dim=0)

        labels = torch.cat([torch.arange(z_i.shape[0]) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z.device)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.t()) / self.temperature

        # discard diagonal
        mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        assert labels.shape == similarity_matrix.shape

        # select and combine multiple positive
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only negatives
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, labels)


class Resnet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_class=10,
                 # planes=(70, 140, 280, 560),
                 planes=(100, 200, 400, 800),
                 dropout=0,
                 bias=False,
                 learning_rate=0.08,
                 lr_min=0.008,
                 weight_decay=0.,
                 devices=None,
                 epochs=150
                 ):
        super(Resnet, self).__init__()
        self.num_class = num_class

        self.input_shortcut_flag = [True]
        self.shortcut_flag = [NO_SHORTCUT]
        self.downsample_flag = [False]
        for i in range(4):
            self.input_shortcut_flag.extend([False, True, False, True])
            self.shortcut_flag.extend([NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, CONCAT_SHORTCUT])
            self.downsample_flag.extend([False, False, False, True])

        self.input_shortcut_flag[-1] = False

        self.layers = nn.ModuleList([
            CWConv(in_channels=in_channels, out_channels=planes[0], kernel_size=3, stride=1, padding=1, bias=bias,
                   num_class=num_class, dropout=dropout),
            # Block 1 (32,32) -> (32,32)
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[0], planes[0], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),

            # Block 2 (32,32) -> (16,16)
            CWConv(planes[1], planes[1], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[1], planes[1], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),

            # Block 3 (16,16) -> (8,8)
            CWConv(planes[2], planes[2], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[2], planes[2], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),

            # CWConv(planes[3], 1000, 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=0),

            # Block 4 (8,8) -> (4,4)
            CWConv(planes[3], planes[3], 3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
            CWConv(planes[3], planes[3], 3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
        ])

        if devices is not None:
            num_layer = len(self.layers)
            num_device = len(devices)
            layer_groups = [i * num_layer // num_device for i in range(num_device)]
            layer_groups.append(num_layer)

            for i in range(num_device):
                for j in range(layer_groups[i], layer_groups[i + 1]):
                    self.layers[j].to(devices[i])

        self.optimizers = [torch.optim.AdamW(layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
                           for layer in self.layers]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
                           for optimizer in self.optimizers]
        # self.schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        #                   for optimizer in self.optimizers]

        # save start_layer and end_layer in parameters
        self.register_buffer('start_layer', torch.tensor(1, dtype=torch.int))
        self.register_buffer('end_layer', torch.tensor(16, dtype=torch.int))


    def forward(self, x, layer_idx=-1, no_norm=False):
        devices = [next(layer.parameters()).device for layer in self.layers]

        g_cls = None
        x = F.layer_norm(x, x.shape[1:])

        shortcut = torch.zeros(1)
        for i, layer in enumerate(self.layers):
            if i > self.end_layer:
                break
            x = x.to(devices[i])

            # return local representation
            if i == layer_idx:
                x, _ = layer(x, no_norm=no_norm)
                return x

            x, g = layer(x)
            if self.start_layer < i <= self.end_layer:
                g_cls += g.to(devices[0])
            elif i == self.start_layer:
                g_cls = g.to(devices[0])

            if self.shortcut_flag[i] == ADD_SHORTCUT:
                x += shortcut.to(x.device)
            elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                x = torch.cat([x, shortcut.to(x.device)], dim=1)

            if self.input_shortcut_flag[i]:
                shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()


        return g_cls

    def update(self, dataloader):
        devices = [next(layer.parameters()).device for layer in self.layers]

        shortcut = torch.zeros(1)
        criterion = nn.CrossEntropyLoss()
        # add InfoNCELoss for representation learning
        contrastive_loss = InfoNCELoss()
        self.train()
        for x, labels in dataloader:
            x, labels = x.to(devices[0]), labels.to(devices[0])
            x = F.layer_norm(x, x.shape[1:])
            for i, layers in enumerate(self.layers):
                x, labels = x.to(devices[i]), labels.to(devices[i])

                self.optimizers[i].zero_grad()
                x, g = layers(x)
                # y, g = layers(x)  # memory saving
                loss = criterion(g, labels)
                loss.backward()
                # alloc_memory = torch.cuda.memory_allocated(device=devices[0])
                self.optimizers[i].step()

                # before_memory = torch.cuda.memory_allocated(device=devices[0])
                # with torch.no_grad(): # memory-saving
                #     del x, g, loss
                #     torch.cuda.empty_cache()
                # after_memory = torch.cuda.memory_allocated(device=devices[0])
                # print(f"Memory: {before_memory / 1024**2:.2f} -> {after_memory/ 1024**2:.2f}")
                # x = y

                # reversed_memory = torch.cuda.memory_reserved(device=devices[0])
                # print(f'Allocated Memory: {alloc_memory / 1024 ** 2:.2f} MB', f' Reserved Memory: {reversed_memory / 1024 ** 2:.2f} MB')

                if self.shortcut_flag[i] == ADD_SHORTCUT:
                    x += shortcut.to(x.device)
                elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                    x = torch.cat([x, shortcut.to(x.device)], dim=1)

                if self.input_shortcut_flag[i]:
                    shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        for scheduler in self.schedulers:
            scheduler.step()


    def pruning(self, dataloader):
        devices = [next(layer.parameters()).device for layer in self.layers]
        corrects = [[0 for _ in range(len(self.layers))] for _ in range(len(self.layers))]
        self.eval()
        total = 0
        with torch.no_grad():
            for x, labels in dataloader:
                shortcut = torch.zeros(1)
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                gs_cls = [0 for _ in range(len(self.layers))]
                for i, layers in enumerate(self.layers):
                    x = x.to(devices[i])
                    x, g = layers(x)
                    for j in range(i + 1):
                        gs_cls[j] = gs_cls[j] + g.to(devices[0])
                        pred = torch.argmax(gs_cls[j], dim=1)
                        corrects[j][i] += torch.eq(pred, labels).sum().float().item()

                    # shortcut
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)

                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

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
        def train_func(module, optimizer, dataloader_size, in_queue, out_queue, in_shortcut_queue, out_shortcut_queue, shortcut_flag):
            device = next(module.parameters()).device
            criterion = nn.CrossEntropyLoss()
            module.train()
            for _ in range(dataloader_size):
                x, labels = in_queue.get()
                x, labels = x.to(device), labels.to(device)

                optimizer.zero_grad()
                x, g = module(x)

                if in_shortcut_queue is not None:
                    shortcut = in_shortcut_queue.get().to(device)
                    if shortcut.shape[2:] != x.shape[2:]:
                        shortcut = F.avg_pool2d(shortcut, 2, stride=2).detach()
                    if shortcut_flag == ADD_SHORTCUT:
                        x += shortcut
                    elif shortcut_flag == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut], dim=1)

                if out_queue is not None:
                    out_queue.put((x, labels))
                if out_shortcut_queue is not None:
                    out_shortcut_queue.put(x.detach())

                loss = criterion(g, labels)
                loss.backward()
                optimizer.step()

        # end of train_func

        self.train()
        main_queues = [queue.Queue(queue_size) for _ in range(len(self.layers))]
        shortcut_queues = [queue.Queue(queue_size) if self.shortcut_flag[i] != NO_SHORTCUT else None
                           for i in range(len(self.layers))]
        main_queues += [None]
        shortcut_queues += [None, None]

        threads = [threading.Thread(target=train_func,
                                    args=(self.layers[i], self.optimizers[i], len(dataloader), main_queues[i],
                                          main_queues[i+1], shortcut_queues[i], shortcut_queues[i+2], self.shortcut_flag[i]))
                   for i in range(len(self.layers))]

        for t in threads:
            t.start()

        for x, labels in dataloader:
            x = x.to(next(self.layers[0].parameters()).device)
            x = F.layer_norm(x, x.shape[1:])
            main_queues[0].put((x, labels))

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
                shortcut = torch.zeros(1)
                x, labels = x.to(devices[0]), labels.to(devices[0])
                total += labels.size(0)
                x = F.layer_norm(x, x.shape[1:])

                for i, layers in enumerate(self.layers):
                    x, labels = x.to(devices[i]), labels.to(devices[i])
                    x, g = layers(x)
                    pred = torch.argmax(g, dim=1)
                    corrects[i] += torch.eq(pred, labels).sum().float().item()

                    # shortcut
                    if self.shortcut_flag[i] == ADD_SHORTCUT:
                        x += shortcut.to(x.device)
                    elif self.shortcut_flag[i] == CONCAT_SHORTCUT:
                        x = torch.cat([x, shortcut.to(x.device)], dim=1)

                    if self.input_shortcut_flag[i]:
                        shortcut = F.avg_pool2d(x, 2, stride=2).detach() if self.downsample_flag[i] else x.detach()

        return [100 * corrects[i] / total for i in range(len(self.layers))]



if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    model = Resnet()
    y = model(x)

