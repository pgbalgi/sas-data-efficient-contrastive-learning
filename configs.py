from enum import Enum
import json

import torchvision
from torchvision.transforms import v2 as transforms
import kornia.augmentation as aug
from kornia.constants import Resample

from collections import namedtuple
from data_proc.augmentation import ColourDistortion
from data_proc.dataset import *
from resnet import *

class SupportedDatasets(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TINY_IMAGENET = "tiny_imagenet"
    IMAGENET = "imagenet"
    STL10 = "stl10"

Datasets = namedtuple('Datasets', 'trainset testset clftrainset num_classes stem')
Transforms = namedtuple('Transforms', 'train test clftrain')

def get_transforms(dataset: str, augment_clf_train=False, kornia=False):

    CACHED_MEAN_STD = {
        SupportedDatasets.CIFAR10.value: ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        SupportedDatasets.CIFAR100.value: ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        SupportedDatasets.STL10.value: ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
        SupportedDatasets.TINY_IMAGENET.value: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        SupportedDatasets.IMAGENET.value: ((0.485, 0.456, 0.3868), (0.2309, 0.2262, 0.2237))
    }

    # Data
    if dataset == SupportedDatasets.STL10.value:
        img_size = 96
    elif dataset == SupportedDatasets.IMAGENET.value:
        img_size = 224
    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
       img_size = 64
    else:
        img_size = 32

    s = 0.5
    if kornia:
        transform_train = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            aug.RandomResizedCrop((img_size,img_size), resample=Resample.BICUBIC),
            aug.RandomHorizontalFlip(),
            aug.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s, p=0.8),
            aug.RandomGrayscale(p=0.2),
            aug.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            ColourDistortion(s=s),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])

    if dataset == SupportedDatasets.IMAGENET.value:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])

    if augment_clf_train:
        if kornia:
            transform_clftrain = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                aug.RandomResizedCrop((img_size,img_size), interpolation=Resample.BICUBIC),
                aug.RandomHorizontalFlip(),
                aug.Normalize(*CACHED_MEAN_STD[dataset]),
            ])
        else:
            transform_clftrain = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(*CACHED_MEAN_STD[dataset]),
            ])
    else:
        transform_clftrain = transform_test

    return Transforms(train=transform_train, clftrain=transform_clftrain, test=transform_test)

def get_datasets(dataset: str, augment_clf_train=False, add_indices_to_data=False, num_positive=2):

    PATHS = {
        SupportedDatasets.CIFAR10.value: '/data/cifar10/',
        SupportedDatasets.CIFAR100.value: '/data/cifar100/',
        SupportedDatasets.STL10.value: '/data/stl10/',
        SupportedDatasets.TINY_IMAGENET.value: '/data/tiny_imagenet/',
        SupportedDatasets.IMAGENET.value: '/data/ILSVRC/'
    }

    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]

    transform_train, transform_clftrain, transform_test = get_transforms(dataset, augment_clf_train)
    trainset = testset = clftrainset = num_classes = stem = None
    
    if dataset == SupportedDatasets.CIFAR100.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR100)
        else:
            dset = torchvision.datasets.CIFAR100
            trainset = CIFAR100Augment(root=root, train=True, download=True, transform=transform_train, n_augmentations=num_positive)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        num_classes = 100
        stem = StemCIFAR

    elif dataset == SupportedDatasets.CIFAR10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10 
            trainset = CIFAR10Augment(root=root, train=True, download=True, transform=transform_train, n_augmentations=num_positive)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        stem = StemCIFAR
    elif dataset == SupportedDatasets.STL10.value:
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STL10
            trainset = STL10Augment(root=root, split='train+unlabeled', download=True, transform=transform_train)
        clftrainset = dset(root=root, split='train', download=True, transform=transform_clftrain)
        testset = dset(root=root, split='test', download=True, transform=transform_test)
        num_classes = 10
        stem = StemSTL

    elif dataset == SupportedDatasets.TINY_IMAGENET.value:
        if add_indices_to_data:
            raise NotImplementedError("Not implemented for TinyImageNet")
        trainset = ImageFolderAugment(root=f"{root}train/", transform=transform_train, n_augmentations=num_positive)  
        clftrainset = ImageFolder(root=f"{root}train/", transform=transform_clftrain)      
        testset = ImageFolder(root=f"{root}test/", transform=transform_train)    
        num_classes = 200
        stem = StemCIFAR
    
    elif dataset == SupportedDatasets.IMAGENET.value:
        if add_indices_to_data:
            raise NotImplementedError("Not implemented for ImageNet")
        trainset = ImageNetAugment(root=f"{root}train_full/", transform=transform_train, n_augmentations=num_positive)
        clftrainset = ImageNet(root=f"{root}train_full/", transform=transform_clftrain)      
        testset = ImageNet(root=f"{root}test/", transform=transform_clftrain)     
        num_classes = 1000
        stem = StemImageNet

    return Datasets(trainset=trainset, testset=testset, clftrainset=clftrainset, num_classes=num_classes, stem=stem)
