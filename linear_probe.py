import argparse
import os
import random
import time

import numpy as np
from sas.subset_dataset import CustomSubsetDataset
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from configs import SupportedDatasets, get_datasets
from evaluate.lbfgs import test_clf
from resnet import *
from util import Random


def main(rank: int, world_size: int, args: int):
    # Determine Device 
    device = rank
    if args.distributed:
        device = args.device_ids[rank]
        torch.cuda.set_device(args.device_ids[rank])
        args.lr *= world_size

    # WandB Logging
    if not args.distributed or rank == 0:
        wandb.init(
            project="data-efficient-contrastive-learning-linear-probe",
            config=args
        )

    if args.distributed:
        ddp_setup(rank, world_size, str(args.port))

    # Set all seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    print('==> Preparing data..')
    datasets = get_datasets(args.dataset)

    testloader = torch.utils.data.DataLoader(
        dataset=CustomSubsetDataset(datasets.testset, subset_indices=range(1000)), 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    clftrainloader = torch.utils.data.DataLoader(
        dataset=datasets.clftrainset, 
        batch_size=args.batch_size, 
        shuffle=not args.distributed,
        sampler=DistributedSampler(CustomSubsetDataset(datasets.clftrainset, subset_indices=range(1000)), shuffle=True) if args.distributed else None,
        num_workers=4, 
        pin_memory=True
    )

    ##############################################################
    # Model and Optimizer
    ##############################################################

    net = torch.load(args.encoder, map_location=device).to(device)

    clf = nn.Linear(net.representation_dim, datasets.num_classes).to(device)
    if args.distributed:
        clf = DDP(clf, device_ids=[device])

    criterion = nn.CrossEntropyLoss()
    clf_optimizer = optim.SGD(clf.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
                            weight_decay=args.weight_decay)

    ##############################################################
    # Encode Data
    ##############################################################

    def encode_data(dataloader):
        all_encodings = torch.zeros((len(dataloader.dataset), net.representation_dim), device=device)
        all_targets = torch.zeros(len(dataloader.dataset), device=device, dtype=torch.int64)

        net.eval()
        t = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs).detach()

            indices = (batch_idx * args.batch_size) + torch.arange(targets.numel())
            all_encodings[indices] = representation
            all_targets[indices] = targets

        return all_encodings, all_targets

    train_encodings, train_targets = encode_data(clftrainloader)
    test_encodings, test_targets = encode_data(testloader)

    ##############################################################
    # Train Function
    ##############################################################

    def train_clf():
        clf.train()
        train_loss = 0
        permutation = torch.randperm(train_targets.size(0))
        for idx in range(0, train_targets.size(0), args.batch_size):
            clf_optimizer.zero_grad()
            indices = permutation[idx:min(idx+args.batch_size, permutation.size(0))]
            representation, targets = train_encodings[indices], train_targets[indices]
            predictions = clf(representation)
            loss = criterion(predictions, targets)
            loss.backward()
            clf_optimizer.step()

            train_loss += loss.item() * targets.size(0)

        train_loss /= train_targets.size(0)
        return train_loss
    
    ##############################################################
    # Test Function
    ##############################################################

    def test_clf():
        clf.eval()
        test_clf_loss = 0
        correct = 0
        acc_per_point = []
        with torch.no_grad():
            for idx in range(0, test_targets.size(0), args.batch_size):
                indices = torch.arange(idx, min(idx+args.batch_size, test_targets.size(0)))
                representation, targets = test_encodings[indices], test_targets[indices]
                raw_scores = clf(representation)
                clf_loss = criterion(raw_scores, targets)
                test_clf_loss += clf_loss.item() * targets.size(0)
                _, predicted = raw_scores.max(1)
                acc_per_point.append(predicted.eq(targets))
                correct += acc_per_point[-1].sum().item()
                
        test_clf_loss /= test_targets.size(0)
        acc = 100. * correct / test_targets.size(0)
        return test_clf_loss, acc

    ##############################################################
    # Main Loop
    ##############################################################     

    # Date Time String
    DT_STRING = str(int(time.time()))

    best_acc = 0
    for epoch in range(args.num_epochs):
        train_loss = train_clf()
        if not args.distributed or rank == 0:
            test_loss, acc = test_clf()
            wandb.log(
                {
                    "test":
                    {
                        "acc": acc,
                        "loss": test_loss,
                        # "top5acc": top5acc
                    },
                    "train":
                    {
                        "loss": train_loss
                    }
                },
                step=epoch
            )
            if acc > best_acc:
                best_acc = acc
                # torch.save(clf, f"{DT_STRING}-{args.dataset}-clf.pt")
            if (epoch + 1) % 10 == 0:
                print('\nEpoch %d' % epoch)
                print('Train Loss: %.3f ' % (train_loss))
                print('Test Loss: %.3f | Test Acc: %.3f%% ' % (test_loss, acc))

    if not args.distributed or rank == 0:
        print("Best test accuracy", best_acc, "%")
        wandb.log(
            {
                "test":
                {
                    "best_acc": best_acc
                }
            }
        )
    
    if args.distributed:
        destroy_process_group()

##############################################################
# Distributed Training Setup
##############################################################
def ddp_setup(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

##############################################################
# Script Entry Point
##############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Linear Probe')
    parser = argparse.ArgumentParser(description='Train downstream classifier with gradients.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
    parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
    parser.add_argument("--num-epochs", type=int, default=2, help='Number of training epochs')
    parser.add_argument("--weight-decay", type=float, default=1e-6, help='Weight decay on the linear classifier')
    parser.add_argument("--nesterov", action="store_true", help="Turn on Nesterov style momentum")
    parser.add_argument("--encoder", type=str, default='ckpt.pth', help='Pretrained encoder')
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])
    parser.add_argument('--device', type=int, default=-1, help="GPU number to use")
    parser.add_argument("--device-ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument('--port', type=int, default=random.randint(49152, 65535), help="free port to use")
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")

    # Parse arguments
    args = parser.parse_args()

    # Arguments check and initialize global variables
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device_ids = None
    distributed = False
    if torch.cuda.is_available():
        if args.device_ids is None:
            if args.device >= 0:
                device = args.device
            else:
                device = "cuda"
        else:
            distributed = True
            device_ids = [int(id) for id in args.device_ids]
    args.device = device
    args.device_ids = device_ids
    args.distributed = distributed
    if distributed:
        mp.spawn(
            fn=main, 
            args=(len(device_ids), args),
            nprocs=len(device_ids)
        )
    else:
        main(device, 1, args)