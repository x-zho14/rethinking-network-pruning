from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import models.cifar as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, ProgressMeter
from torch.utils.tensorboard import SummaryWriter
from utils.misc import get_conv_zero_param


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
parser.add_argument('--clipping', default=False, action="store_true", help="whether use gradient clipping")
parser.add_argument('--number', type=int, default=0, help='log number')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print(model)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # # Load checkpoint.
        # print('==> Resuming from checkpoint..')
        # assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.save_dir = os.path.dirname(args.resume)
        # checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        # start_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title, resume=True)

        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        # logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        print("else")

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    log_base_dir = "logs" + str(args.number)
    # Train and val
    writer = SummaryWriter(log_dir=log_base_dir)
    for epoch in range(start_epoch, args.epochs):
        # newlr = adjust_learning_rate(optimizer, epoch)

        # print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, newlr))
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        num_parameters = get_conv_zero_param(model)
        print('Zero parameters: {}'.format(num_parameters))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, writer)

        # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir)

    # logger.close()

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()

    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(trainloader), [losses, top1, top5], prefix="Train: "
    )

    # bar = Bar('Processing', max=len(trainloader))
    # print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        optimizer.zero_grad()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        loss.backward()
        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        if args.clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)


        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 100 == 0:
            progress.display(batch_idx)
    progress.display(len(trainloader))
    progress.write_to_tensorboard(writer, prefix="train", global_step=epoch)
        # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(trainloader),
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 top1=top1.avg,
    #                 top5=top5.avg,
    #                 )
    #     bar.next()
    # bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, writer):
    global best_acc

    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(testloader), [losses, top1, top5], prefix="Test: "
    )


    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 100 == 0:
                progress.display(batch_idx)
        progress.display(len(testloader))
        progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
            # plot progress
    #         bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                     batch=batch_idx + 1,
    #                     size=len(testloader),
    #                     data=data_time.avg,
    #                     bt=batch_time.avg,
    #                     total=bar.elapsed_td,
    #                     eta=bar.eta_td,
    #                     loss=losses.avg,
    #                     top1=top1.avg,
    #                     top5=top5.avg,
    #                     )
    #         bar.next()
    # bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    newlr = 0.5 * (1 + np.cos(np.pi * epoch / state['epochs'])) * state['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = newlr
    return newlr
    # if epoch in args.schedule:
    #     state['lr'] *= args.gamma
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
