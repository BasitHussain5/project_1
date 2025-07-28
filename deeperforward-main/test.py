import time
import os, sys

import torch
# DataParallel
from torch.nn.parallel import DataParallel

from data.cifar10_dataloader import get_cifar10_dataloader
from data.mnist_dataloader import get_mnist_dataloader
from data.fmnist_dataloader import get_fmnist_dataloader

from model.alexnet import Alexnet
from model.noresidual import NoResnet
from model.vgg import Vgg
from model.resnet import Resnet
from model.tinycnn import TinyCNN


def test_model(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    for x, labels in test_loader:
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            y = model(x)
            pred = torch.argmax(y, dim=1)
            correct += torch.eq(pred, labels).sum().float().item()
        total += labels.size(0)
    test_acc = 100 * correct / total
    return test_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch DeepForward Testing')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--task_dir', type=str, default=None)
    parser.add_argument('--save_result', action='store_true', default=False)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--nopruning', action='store_true', default=False)

    args = parser.parse_args()

    task = args.task
    arch = args.arch
    task_dir = os.path.join('./results', args.task_dir)
    save_result = args.save_result
    seed = args.seed
    root = args.data_dir

    devices = [torch.device('cuda:{}'.format(device)) for device in args.devices]

    max_layer = 0

    # error
    if task_dir is None:
        print('Please specify task_dir by --task_dir')
        sys.exit(1)

    model_dir = os.path.join(task_dir, 'model.pth')
    if not os.path.exists(model_dir):
        print('No model.pth in {}'.format(task_dir))
        sys.exit(1)

    in_channels = 3
    num_class = 10
    dropout = 0.2
    lr = 0.08
    lr_min = 0.008
    epochs = 150
    weight_decay = 1e-4
    train_loader, valid_loader, test_loader = None, None, None
    if task == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloader(root=root, train_batch_size=128,
                                                                         test_batch_size=128, seed=seed)
    elif task == 'fmnist':
        in_channels = 1
        train_loader, valid_loader, test_loader = get_fmnist_dataloader(root=root, train_batch_size=128,
                                                                        test_batch_size=128, seed=seed)
    elif task == 'mnist':
        in_channels = 1
        train_loader, valid_loader, test_loader = get_mnist_dataloader(root=root, train_batch_size=128,
                                                                       test_batch_size=128, seed=seed)

        # model
    model = None
    if args.arch == 'alexnet':
        model = Alexnet(in_channels=in_channels, num_class=num_class, dropout=dropout)
    elif args.arch == 'vgg':
        model = Vgg(in_channels=in_channels, num_class=num_class, dropout=dropout, learning_rate=lr, lr_min=lr_min,
                    devices=devices)
    elif args.arch == 'resnet':
        model = Resnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                       planes=[100, 200, 400, 800], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                       devices=devices, epochs=epochs)
    elif args.arch == 'tinycnn':
        model = TinyCNN(in_channels, num_class=num_class, dropout=dropout, learning_rate=lr, devices=devices,
                        epochs=epochs,
                        lr_decay=lr_min / lr)
    elif args.arch == 'noresnet':
        model = NoResnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                         planes=[100, 200, 400, 800], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                         devices=devices)
    elif args.arch == 'resnet_large':
        model = Resnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                       planes=[300, 600, 1200, 2400], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                       devices=devices, epochs=epochs)

    # load model
    model.load_state_dict(torch.load(model_dir), strict=False)
    model = model.to(devices[0])

    if args.nopruning:
        model.start_layer = 0
        model.end_layer = max_layer

    model.eval()
    train_acc = test_model(model, train_loader, devices[0])
    test_acc = test_model(model, test_loader, devices[0])
    print('Train Acc: {:.2f}%, Test Acc: {:.2f}%'.format(train_acc, test_acc))
    print('Start layer: {}, End layer: {}'.format(model.start_layer, model.end_layer))

    if save_result:
        train_local_acc = model.test_local_acc(train_loader)
        test_local_acc = model.test_local_acc(test_loader)

        train_layer_acc_list = model.test_local_acc(train_loader)
        test_layer_acc_list = model.test_local_acc(test_loader)

    if save_result:
        with open(os.path.join(task_dir, 'layer_acc.csv'), 'w') as f:
            for i in range(len(train_layer_acc_list)):
                f.write(f'{train_layer_acc_list[i]},{test_layer_acc_list[i]}\n')
        print('Save layer_acc.csv to {}'.format(task_dir))
