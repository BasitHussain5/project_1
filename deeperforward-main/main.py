import time
import os, sys

import torch
# DataParallel
from torch.nn.parallel import DataParallel

from data.cifar10_dataloader import get_cifar10_dataloader
from data.mnist_dataloader import get_mnist_dataloader
from data.fmnist_dataloader import get_fmnist_dataloader
from data.cifar100_dataloader import get_cifar100_dataloader

from model.alexnet import Alexnet
from model.noresidual import NoResnet
from model.vgg import Vgg
from model.resnet import Resnet
from model.tinycnn import TinyCNN
from model.resnet101 import Resnet101
from model.resnet20 import Resnet20
from model.resnet34 import Resnet34

from thop import profile, clever_format

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

# torch.cuda.set_per_process_memory_fraction(0.05, 0)
# torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch DeepForward Training')
    parser.add_argument('--task', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.08)
    parser.add_argument('--lr_min', type=float, default=0.008)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--task_dir', type=str, default=None)
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--save_step', type=int, default=5)
    parser.add_argument('--queue_size', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    lr_min = args.lr_min
    weight_decay = args.weight_decay
    dropout = args.dropout

    if args.task_dir is None:
        task_dir = os.path.join('./results/', f'{time.strftime("%Y%m%d-%H%M%S")}_{args.task}_{args.arch}')
    else:
        task_dir = os.path.join('./results/', args.task_dir)

    if task_dir is not None:
        # create directory if not exists
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

    in_channels = 3
    num_class = 10
    start_epoch = 0

    root = args.data_dir
    img_size = 32

    devices = []
    for i in args.devices:
        devices.append(torch.device('cuda:{}'.format(i) if torch.cuda.is_available() else 'cpu'))

    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    checkpoint_path = './checkpoint/checkpoint.pth'
    if task_dir is not None:
        checkpoint_path = os.path.join(task_dir, 'checkpoint.pth')

    task = args.task
    train_loader, valid_loader, test_loader = None, None, None
    if task == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloader(root=root, train_batch_size=train_batch_size,
                                                                         test_batch_size=test_batch_size, seed=seed)
    elif task == 'fmnist':
        in_channels = 1
        train_loader, valid_loader, test_loader = get_fmnist_dataloader(root=root, train_batch_size=train_batch_size,
                                                                        test_batch_size=test_batch_size, seed=seed)
    elif task == 'mnist':
        in_channels = 1
        train_loader, valid_loader, test_loader = get_mnist_dataloader(root=root, train_batch_size=train_batch_size,
                                                                       test_batch_size=test_batch_size, seed=seed)
    elif task == 'cifar100':
        num_class = 100
        train_loader, valid_loader, test_loader = get_cifar100_dataloader(root=root, train_batch_size=train_batch_size,
                                                                          test_batch_size=test_batch_size, seed=seed)

    # model
    model = None
    if args.arch == 'alexnet':
        model = Alexnet(in_channels=in_channels, num_class=num_class, dropout=dropout)
    elif args.arch == 'vgg':
        model = Vgg(in_channels=in_channels, num_class=num_class, dropout=dropout, learning_rate=lr, lr_min=lr_min, devices=devices)
    elif args.arch == 'resnet':
        model = Resnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                       planes=[100,200,400,800], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                       devices=devices, epochs=epochs)
    elif args.arch == 'tinycnn':
        model = TinyCNN(in_channels, num_class=num_class, dropout=dropout, learning_rate=lr, devices=devices, epochs=epochs,
                        lr_decay=lr_min/lr)
    elif args.arch == 'noresnet':
        model = NoResnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                         planes=[100,200,400,800], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                         devices=devices)
    elif args.arch == 'resnet_large':
        model = Resnet(in_channels=in_channels, num_class=num_class, dropout=dropout,
                       planes=[300, 600, 1200, 2400], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                       devices=devices, epochs=epochs)
    elif args.arch == 'resnet101':
        model = Resnet101(in_channels=in_channels, num_class=num_class, dropout=dropout,
                          planes=[70, 140, 280, 560], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                          devices=devices, epochs=epochs)
    elif args.arch == 'resnet20':
        model = Resnet20(in_channels=in_channels, num_class=num_class, dropout=dropout,
                         planes=[100, 200, 400], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                         devices=devices, epochs=epochs)
    elif args.arch == 'resnet34':
        model = Resnet34(in_channels=in_channels, num_class=num_class, dropout=dropout,
                         planes=[100, 200, 400, 800], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                         devices=devices, epochs=epochs)
    elif args.arch == 'resnet34_large':
        model = Resnet34(in_channels=in_channels, num_class=num_class, dropout=dropout,
                         planes=[300, 600, 1200, 2400], learning_rate=lr, lr_min=lr_min, weight_decay=weight_decay,
                         devices=devices, epochs=epochs)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        start_epoch += 1
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for i, optimizer in enumerate(model.optimizers):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][i])
        for i, scheduler in enumerate(model.schedulers):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'][i])
        print(f'Load checkpoint from {checkpoint_path} start from epoch {start_epoch + 1}')

    input_data = torch.randn(1,3,32,32, device=devices[0])
    flops, params = profile(model, inputs=(input_data,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'FLOPs: {flops}, Params: {params}')


    train_acc, valid_acc, test_acc = 0., 0., 0.
    for epoch in range(start_epoch, epochs):

        start_time = time.time()
        model.train()
        if args.parallel:
            model.update_pipeline(train_loader, queue_size=args.queue_size)
        else:
            model.update(train_loader)
        model.eval()
        train_time = time.time() - start_time

        # pruning
        pruning_time = time.time()
        valid_acc, before_acc = model.pruning(valid_loader)
        pruning_time = time.time() - pruning_time

        test_time = time.time()
        test_acc = test_model(model, test_loader, devices[0])
        test_time = time.time() - test_time

        if epoch % args.save_step == 0:
            torch.save(model.state_dict(), checkpoint_path)
            train_acc = test_model(model, train_loader, devices[0])
            end_time = time.time() - start_time

            info = f'Epoch: {(epoch + 1):03d}/{epochs:03d}: ' \
                   f'Train Acc: {train_acc:.2f}% \t || Test training-set Time: {end_time:.2f}s'
            print(info)

            # saving the accuracy of training and testing into csv file
            if task_dir is not None:
                with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
                    f.write(f'{epoch},{train_acc},{valid_acc},{test_acc}\n')

        info = f'Epoch: {(epoch + 1):03d}/{epochs:03d}: ' \
               f'Pruning ({model.start_layer:02d}->{model.end_layer:02d}):\t' \
               f'Valid Acc:{before_acc:.2f}% -> {valid_acc:.2f}%\t ' \
               f'Test Acc: {test_acc:.2f}% \t || ' \
               f'Train Time: {train_time:.2f}s, ' \
               f'Pruning Time: {pruning_time:.2f}s '\
               f'Test Time: {test_time:.2f}s'\
               f'|| lr: {model.optimizers[0].param_groups[0]["lr"]:.5f}'
        print(info)

        # checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': [optimizer.state_dict() for optimizer in model.optimizers],
            'scheduler_state_dict': [scheduler.state_dict() for scheduler in model.schedulers],
        }
        torch.save(checkpoint, checkpoint_path)

    # Finishing training

    start_time = time.time()
    train_acc = test_model(model, train_loader, devices[0])
    end_time = time.time() - start_time

    print(f'Final: Train Acc: {train_acc:.2f}% \t || Test training-set Time: {end_time:.2f}s')
    if task_dir is not None and args.save_step != 1:
        with open(os.path.join(task_dir, 'accuracy.csv'), 'a') as f:
            f.write(f'{epochs},{train_acc},{valid_acc},{test_acc}\n')

    train_layer_acc_list = model.test_local_acc(train_loader)
    test_layer_acc_list = model.test_local_acc(test_loader)
    if task_dir is not None:
        with open(os.path.join(task_dir, 'layer_acc.csv'), 'w') as f:
            for i in range(len(train_layer_acc_list)):
                f.write(f'{train_layer_acc_list[i]},{test_layer_acc_list[i]}\n')

    model = model.to('cpu')
    if task_dir is not None:
        model_path = os.path.join(task_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
    # remove checkpoint
    os.remove(checkpoint_path)
