import torch
from torchvision import datasets, transforms
import numpy as np

def get_splits():

    # GET THE TRAINING DATASET
    train_data = datasets.MNIST(root='MNIST-data',                        # where is the data (going to be) stored
        transform=transforms.ToTensor(),          # transform the data from a PIL image to a tensor
        train=True,                               # is this training data?
        download=True                             # should i download it if it's not already here?
    )

    # GET THE TEST DATASET
    test_data = datasets.MNIST(root='MNIST-data',
        transform=transforms.ToTensor(),
        train=False,
    )

    train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])    # split into 50K training & 10K validation
    
    return train_data, val_data, test_data


def get_dataloaders(batch_size=16):

    train_data, val_data, test_data = get_splits()

    # MAKE TRAINING DATALOADER
    train_loader = torch.utils.data.DataLoader( # create a data loader
        train_data, # what dataset should it sample from?
        shuffle=True, # should it shuffle the examples?
        batch_size=batch_size # how large should the batches that it samples be?
    )

    # MAKE VALIDATION DATALOADER
    val_loader = torch.utils.data.DataLoader(
        val_data,
        shuffle=True,
        batch_size=batch_size
    )

    # MAKE TEST DATALOADER
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader

def test_accuracy(net, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# test_acc = test_accuracy(best_trained_model, device)
# print(f"Best trial test set accuracy: ({test_acc*100}%) achieved with {best_trial.config['n_layers']} layers")
