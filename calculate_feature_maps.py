import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import cifar10, imagenet
import time
import random
from models.resnet_cifar10 import resnet_56,resnet_110
from models.resnet_imagenet import resnet_50
from models.vgg_cifar10 import vgg_16_bn
from models.mobilenetv2 import mobilenet_v2
import matplotlib.pyplot as plt
import torchvision

parser = argparse.ArgumentParser(description='Calculate Feature Maps')

parser.add_argument(
    '--arch',
    type=str,
    default='mobilenet_v2',
    choices=('vgg_16_bn','resnet_56','mobilenet_v1','resnet_50','mobilenet_v2'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--dataset',
    type=str,
    default='imagenet',
    choices=('cifar10','imagenet'),
    help='cifar10 or imagenet')

parser.add_argument(
    '--data_dir',
    type=str,
    default='/data1/wanghx/dataset/ImageNet',
    help='dataset path')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./resnet50.pth',
    help='dir for the pretriained model to calculate feature maps')

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='batch size for one batch.')
parser.add_argument(
    "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
)
parser.add_argument(
    "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
)
parser.add_argument(
    "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
)
parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='the number of different batches for calculating feature maps.')

parser.add_argument(
    '--gpu',
    type=str,
    default='5',
    help='gpu id')

args = parser.parse_args()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def worker_init_fn(work_id):
    worker_seed = torch.initial_seed() %2**32
    np.random.seed(worker_seed+work_id)

# gpu setting
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_seed(1234)
# prepare data
if args.dataset=='cifar10':
    trainset, testset = cifar10.load_cifar_data(args)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               worker_init_fn=worker_init_fn)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    train_loader = torch.utils.data.DataLoader(data_tmp.trainset, batch_size=args.batch_size, shuffle=True,num_workers=0, worker_init_fn=worker_init_fn,pin_memory=True)


# Model
model = eval(args.arch)(sparsity=[0.]*100).to(device)
print(model)

# Load pretrained model.
print('Loading Pretrained Model...')
if args.arch=='vgg_16_bn' or args.arch=='resnet_56':
    checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)
else:
    checkpoint = torch.load(args.pretrain_dir)
if args.arch=='resnet_50':
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint['state_dict'])

conv_index = torch.tensor(1)
print('Loading Pretrained Model finished...')
def get_feature_hook(self, input, output):
    global conv_index

    if not os.path.isdir('conv_feature_map/' + args.arch + '_repeat%d' % (args.repeat)):
        os.makedirs('conv_feature_map/' + args.arch + '_repeat%d' % (args.repeat))
    np.save('conv_feature_map/' + args.arch + '_repeat%d' % (args.repeat) + '/conv_feature_map_'+ str(conv_index) + '.npy',
            output.cpu().numpy())
    conv_index += 1

def inference():
    model.eval()
    repeat = args.repeat
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use 5 batches to get feature maps.
            if batch_idx >= repeat:
               break

            inputs, targets = inputs.to(device), targets.to(device)

            model(inputs)

if args.arch=='vgg_16_bn':

    if len(args.gpu) > 1:
        relucfg = model.module.relucfg
    else:
        relucfg = model.relucfg
    start = time.time()

    for i, cov_id in enumerate(relucfg):
        cov_layer = model.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

elif args.arch=='resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1


elif args.arch=='resnet_50':
    cov_layer = eval('model.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet50 per bottleneck
    for i in range(4):
        block = eval('model.layer%d' % (i + 1))
        for j in range(model.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            if j==0:
                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
elif args.arch == 'mobilenet_v2':
    cov_layer = eval('model.features[0]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    for i in range(1, 19):
        if i == 1:
            block = eval('model.features[%d].conv' % (i))
            relu_list = [2, 4]
        elif i == 18:
            block = eval('model.features[%d]' % (i))
            relu_list = [2]
        else:
            block = eval('model.features[%d].conv' % (i))
            relu_list = [2, 5, 7]

        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()


elif args.arch == 'mobilenet_v1':
    cov_layer = eval('model.conv1[2]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    for i in range(13):
        block = eval('model.features[%d]' % (i))
        relu_list = [2, 5]
        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1
else:
    raise NotImplementedError