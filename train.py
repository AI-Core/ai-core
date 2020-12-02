import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from ray import tune
import os
# def train(train_epoch, model):
#     # , epochs=10):

#     writer = SummaryWriter(log_dir=f'runs/Autoencoder-{time()}')
#     batch_idx = 0
#     optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
#     for epoch in range(epochs):
#         train_epoch(model)

def checkpoint(model, optimiser, epoch):
    # def wrapper(*args):
    #     print('making checkpoint')
    #     train_epoch(*args)

    print(model)
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimiser.state_dict()), path)
        # wrapper.epoch += 1

    # wrapper.epoch = 0
    # return wrapper

# @checkpoint
# def train_epoch(batch_loss):
#     """Wraps around a function which trains a batch"""

#     # train_epoch.batch_idx = 0

#     def wrapper(model, epoch):
#         for batch in train_loader:
#             loss = batch_loss(model, batch)
#             loss.backward()
#             optimiser.step()
#             optimiser.zero_grad()
#             writer.add_scalar('Autoeocoder/Loss', loss, wrapper.batch_idx)
#             # batch_idx += 1
#             wrapper.batch_idx += 1

#         checkpoint(model, optimiser, epoch)

#     wrapper.batch_idx = 0

#     return wrapper

def validate(model, device, val_loader, batch_idx, loss_fn, writer):
    model.eval()
    print('validating')
    val_loss = 0
    for (x, y) in val_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    writer.add_scalar(f'{writer.logdir}/Loss/Validation', val_loss, batch_idx)
    print(batch_idx)
    print(batch_idx, val_loss)
    model.train()

def train(model, logdir, train_loader, val_loader, test_loader, loss_fn, epochs=1, on_epoch_end=None, verbose=False):
    model.train()
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(device)
    model = model.to(device)

    writer = SummaryWriter(log_dir=f'runs/{logdir}-{time()}')
    writer.logdir = logdir
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_idx = 0
    # validate(model, val_loader, batch_idx, loss_fn, writer)
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            # loss = batch_loss(model, batch)
            pred = model(x)
            # print(pred.shape)
            # print(y.shape)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar(f'{writer.logdir}/Loss/Train', loss.item(), batch_idx)
            batch_idx += 1
            if verbose:
                print(f'Epoch: {epoch}\tBatch: {batch_idx}\tLoss: {loss.item()}')
        validate(model, device, val_loader, batch_idx, loss_fn, writer)
        if on_epoch_end:
            on_epoch_end(model, writer, device, epoch)

        # checkpoint(model, optimiser, epoch)
    return model, writer
