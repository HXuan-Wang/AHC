'''
https://github.com/lmbxmu/HRankPlus
'''
import torch
import torch.utils
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms

import argparse

def load_cifar_data(args):

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


    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                            transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)


    return trainset, testset

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CIFAR-10 training")

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./',
        help='path to dataset')
    args = parser.parse_args()
    train_dataset, val_dataset = load_cifar_data(args)
    print('data_size:', len(train_dataset))
    print('data_class:', (train_dataset.classes))
    print('class_id:', (train_dataset.class_to_idx))