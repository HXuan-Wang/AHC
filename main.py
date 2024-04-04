import os
import numpy as np
import time, datetime
import torch
import argparse
import math
import shutil
from collections import OrderedDict
import torchvision.models
from thop import profile
from scaler import NativeScaler_part_update
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from logger import create_logger
from data import imagenet
import utils


from models.resnet_imagenet import resnet_50
from models.mobilenetv2 import mobilenet_v2
from models.mobilenetv1 import mobilenet_v1

parser = argparse.ArgumentParser("ImageNet training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='',
    help='path to dataset')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_50',
    help='architecture')

parser.add_argument(
    '--result_dir',
    type=str,
    default='./result',
    help='results path for saving models and loggers')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=90,
    help='num of training epochs')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='init learning rate')

parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

parser.add_argument(
    '--lr_type',
    default='cos',
    type=str,
    help='learning rate decay schedule')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='weight decay')

parser.add_argument(
    '--label_smooth',
    type=float,
    default=0.1,
    help='label smoothing')
parser.add_argument(
    "--norm-weight-decay",
    default=None,
    type=float,
    help="weight decay for Normalization layers (default: None, same value as --wd)",
)
parser.add_argument('--log', type=str, default='same',
                        help='log file path')
parser.add_argument('--log_period', type=int, default=10,
                        help='log period in training or finetuning')

parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
parser.add_argument(
    "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
)
parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument(
    "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
)
parser.add_argument(
    "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
)
parser.add_argument(
    "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
)
parser.add_argument(
    "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
)
parser.add_argument("--lr_min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='',
    help='pretrain model path')
parser.add_argument('--seed', default=0, type=int)

parser.add_argument(
    '--data_set',
    type=str,
    default='ImageNet-few',
    help='architecture')
parser.add_argument(
    '--milestones',
    nargs='+',
    help='parameters of PDP')
parser.add_argument(
    '--data_pruning_ratio',
    default=0.4,
    type=float,
    help='parameters of PDP')
parser.add_argument(
    '--lambada',
    default=0.2,
    type=float,
    help='parameters of lambada')

args = parser.parse_args()

#use for loading pretrain model
name_base=''

def main(args):

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.log == 'same':
        args.log = args.result_dir
    log_file = os.path.join(args.log, '{}.log'.format(timestamp))
    logger = create_logger(output_dir=log_file)

    args_txt = '----------running configuration----------\n'
    for key, value in vars(args).items():
        args_txt += ('{}: {} \n'.format(key, str(value)))
    logger.info(args_txt)

    logger.info('==> Building model..')
    model = torch.load(args.pretrain_dir)
    logger.info(model)

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    print('==> Preparing data..')

    if args.data_set == 'ImageNet':
        data_tmp = imagenet.Data(args)
        dataset_train = data_tmp.trainset
        dataset_val = data_tmp.testset
    else:
        raise NotImplementedError


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    criterion = nn.MSELoss(reduction='mean')
    criterion = criterion.to(device)

    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = criterion_eval.to(device)

    best_top1_acc= 0
    best_top5_acc= 0

    if args.arch=='resnet_50':
        origin_model = resnet_50(sparsity=[0.]*100)
        ckpt = torch.load('./resnet50.pth', map_location='cpu')
        origin_model.load_state_dict(ckpt)
    elif args.arch == 'mobilenet_v1':
        origin_model = mobilenet_v1(sparsity=[0.] * 100)
        ckpt = torch.load('./mobilenet_v1.pth.tar', map_location='cpu')
        origin_model.load_state_dict(ckpt)
    elif args.arch=='mobilenet_v2':
        from torchvision.models import MobileNet_V2_Weights
        origin_model = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        raise NotImplementedError

    origin_model.to(device)

    model.to(device)

    loss_scaler = NativeScaler_part_update()
    start_t = time.time()
    is_best = False

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    milestones = [int(x) for x in args.milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    selected_data = dataset_train
    for epoch in range(0, int(args.epochs)):
        if epoch in milestones:
            data_loader_train = torch.utils.data.DataLoader(selected_data, batch_size=args.batch_size,
                                                            num_workers=32,
                                                            pin_memory=args.pin_mem,
                                                            drop_last=False,
                                                            shuffle=False)
            selected_data = Progressive_data_pruning(epoch, data_loader_train, model, origin_model, args.lambada,args.data_pruning_ratio,selected_data, optimizer, loss_scaler, device, logger, args.clip_grad)

            data_loader_train = torch.utils.data.DataLoader(selected_data,batch_size=args.batch_size,
            num_workers=32,
            pin_memory=args.pin_mem,
            drop_last=True,
            shuffle=True)
        else:
            i=0
            train_obj, train_top1_acc, train_top5_acc = Fine_tuning(epoch, data_loader_train, model,
                                                                               origin_model, criterion, criterion_eval,
                                                                               optimizer, i, loss_scaler, device,
                                                                               logger,
                                                                               args.clip_grad)
        lr_scheduler.step()
        if (epoch) %10==0:
            valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, data_loader_val, model, criterion_eval, device, logger)
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                best_top5_acc = valid_top5_acc
                is_best = True

            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'best_top5_acc': best_top5_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.result_dir)

    logger.info("=>Best accuracy Top1: {:.3f}, Top5: {:.3f}".format(best_top1_acc, best_top5_acc))

    training_time = (time.time() - start_t) / 36000
    logger.info('total training time = {} hours'.format(training_time))

def Progressive_data_pruning(epoch, train_loader, model, model_t, lambada, ratio, selected_data,optimizer, scaler, device,logger, max_norm: float = 0):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    l1_losses = utils.AverageMeter('L1 Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.eval()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = 50

    model_t.eval()
    criterion = nn.MSELoss(reduction='none').to(device)
    criterion_eval = nn.CrossEntropyLoss().to(device)
    losses_with_indices = []

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast():

            logits = model(images)

            with torch.no_grad():
                logits_t = model_t(images)

            loss_ce = criterion_eval(logits, targets)
            loss_l1 = criterion(logits, logits_t)
            loss = loss_l1 + lambada * loss_ce
            total_loss_mean = torch.mean(loss,dim=1)
            total_loss = loss.mean()
            losses_with_indices.extend(total_loss_mean.cpu().detach().numpy())

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)

        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # # compute gradient and do SGD step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        scaler(total_loss, optimizer, clip_grad=max_norm,
                    update=False,
                    parameters=model.parameters(), create_graph=is_second_order)


        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter,
                    top1=top1, top5=top5))
    sample_value = torch.tensor(losses_with_indices)

    sample_value = sample_value.to(device)

    sorted_data, indices = torch.sort(sample_value)

    selected_indices = indices[-int(len(indices) * ratio):]

    next_selected_data = [selected_data[idx] for idx in selected_indices]

    return next_selected_data


def Fine_tuning(epoch, train_loader, model, model_t,criterion, criterion_eval,optimizer, lambada, scaler, device,logger, max_norm: float = 0):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    l1_losses = utils.AverageMeter('L1 Loss', ':.4e')
    ce_losses = utils.AverageMeter('CE Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')


    model.train()
    end = time.time()
    # scheduler.step()

    print_freq = 50

    num_iter = len(train_loader)

    model_t.eval()

    for batch_idx, (images, targets) in enumerate(train_loader):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast():
            logits = model(images)
            with torch.no_grad():
                logits_t = model_t(images)
            loss_ce = criterion_eval(logits,targets)
            loss_l1 = criterion(logits, logits_t)

            loss = loss_l1+lambada*loss_ce

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        l1_losses.update(loss_l1.item(), n)  # accumulated loss
        ce_losses.update(loss_ce.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        scaler(loss, optimizer, clip_grad=max_norm,
               update=True,
               parameters=model.parameters(), create_graph=is_second_order)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'L1_Loss {l1_loss.avg:.4f} '
                'CE_Loss {ce_loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, l1_loss=l1_losses,ce_loss= ce_losses,
                    top1=top1, top5=top5))

    return ce_losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, device,logger):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    args = parser.parse_args()
    os.umask(0)
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
    os.umask(0)
    main(args)
