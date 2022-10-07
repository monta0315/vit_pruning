import argparse
import csv
import os
from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#from models.slim_split import ViT_slim as ViT
#from models.slim_split import channel_selection
from models.select_split import ViT, channel_selection
from utils.utils import progress_bar

# parsers
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--lr", default=1e-4, type=float, help="learning rate"
)  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument("--opt", default="adam")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--aug", action="store_true", help="add image augumentations")
parser.add_argument("--mixup", action="store_true", help="add mixup augumentations")
parser.add_argument("--net", default="vit")
parser.add_argument("--bs", default="64")
parser.add_argument("--n_epochs", type=int, default="100")
parser.add_argument("--patch", default="4", type=int)
parser.add_argument(
    "--cos", action="store_true", help="Train with cosine annealing scheduling"
)
args = parser.parse_args()


if args.cos:
    from warmup_scheduler import GradualWarmupScheduler
if args.aug:
    import albumentations

bs = int(args.bs)

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0
start_epoch = 0

print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

if not os.path.isdir("data"):
    os.makedirs("data")

# download dataset
trainset = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform_train
)
# define dataloader
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bs, shuffle=True, num_workers=8
)

testset = torchvision.datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8
)

# declare Cifar10 Classes
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model phase

print("==> Building model..")

net = ViT(
    image_size=32,
    patch_size=args.patch,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
)

net = net.to(device)


# if use pretraind model
if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/{}-ckpt.t7".format(args.net))
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

# declare Lodd and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)

from torch.optim import lr_scheduler

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, verbose=True, min_lr=1e-3 * 1e-5, factor=0.1
)


def sparce_selection():
    s = 1e-4
    for m in net.modules():
        if isinstance(m, channel_selection):
            """
                >>> a
                tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
                >>> torch.sign(a)
                tensor([ 1., -1.,  0.,  1.])
            """
            m.indexes.grad.data.add_(s * torch.sign(m.indexes.data))  # 1か0か-1に丸め込んでる


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sparce_selection()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
    return train_loss / (batch_idx + 1)


### validation
import time


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    if not args.cos:
        scheduler.step(test_loss)

    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving")
        state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}

        if not os.path.isdir("checkpoint"):
            os.makedirs("checkpoint")

        torch.save(
            state,
            f"./checkpoint/{args.net}-CIFAR10-{args.n_epochs}epochs-{args.bs}bs.pth".format(
                args.patch
            ),
        )
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = (
        time.ctime()
        + " "
        + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    )
    print(content)
    with open(f"log/log_{args.net}_patch{args.patch}.txt", "a") as appender:
        appender.write(content + "\n")
    return test_loss, acc


list_loss = []
list_acc = []

for epoch in range(start_epoch, args.n_epochs):
    trainloss = train(epoch)
    val_loss, acc = test(epoch)

    list_loss.append(val_loss)
    list_acc.append(acc)

    with open(f"log/log_{args.net}_patch{args.patch}.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list_loss)
        writer.writerow(list_acc)
