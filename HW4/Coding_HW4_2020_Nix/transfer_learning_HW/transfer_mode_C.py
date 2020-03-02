import torch
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from torchvision import models
from torchvision.transforms import FiveCrop, ToTensor, Lambda, Compose, CenterCrop, Normalize

import pandas as pd
import numpy as np
import torch.nn as nn

from torch.nn import CrossEntropyLoss

from os import listdir
from os.path import isfile, join

import time

import matplotlib.pyplot as plt

class FlowersDataset(Dataset):
    def __init__(self, img_root, ins_label_pairs , crop_size, transform=None):
        
        """
        
        img_root: contains the path to the image root folder
        ins_label_pairs: contains a list of all the image path names and their respective labels
        crop_size: contains desired crop dimensions
        transform: contains the transformation procedures to be applied. defaulted to be None
        
        """
        self.img_root = img_root
        self.ins_label_pairs = ins_label_pairs
        self.crop_size = crop_size
        self.transform = transform
        
    def __len__(self):
        return len(self.ins_label_pairs)
    
    def image_load(self, image_path):
        # Open image and load
        img = Image.open(image_path)
        img.load()
        
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
            
        return Image.fromarray(img)
    
    def image_resize(self, image, crop_size):
        W, H = image.size
        # Scale according to the lower value between height and width
        scale = crop_size / min(W, H)
        # New size for resizing
        new_size = (int(np.ceil(scale * W)), int(np.ceil(scale * H)))
        
        return image.resize(new_size)
        
    def __getitem__(self, index):
        # Path to the image
        image_path = self.img_root + self.ins_label_pairs[index][0]
        
        # Open the image
        image = self.image_load(image_path)
        label = self.ins_label_pairs[index][1]
        
        if self.transform is not None:
            image = self.image_resize(image, self.crop_size)
            image = self.transform(image)
            
        return [image, label]

def train(model, train_loader, criterion, device, optimizer):
    """
    model: your model to train
    train_loader: your train dataloader
    criterion: your loss function
    device: the device you are computing with
    optimizer: the one updating your weights
    """
    model.train()
    
    # To store your train losses for later
    losses = []
    
    for idx, batch in enumerate(train_loader):
        inputs, labels = batch[0].to(device), batch[1].long().to(device)
        optimizer.zero_grad()
        
        # Get your outputs
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        # Back propagate
        loss.backward()
        
        # Update
        optimizer.step()
        
        losses.append(loss.item())
        
    losses = np.average(losses)
    return losses

def evaluate(model, data_loader, criterion, device):
    """
    model: your model to train
    data_loader: your validation or test dataloader
    criterion: your loss function
    device: the device you are computing with
    
    """
    model.eval()
    
    # To get your accuracy later
    running_corrects = 0
    
    # Store your losses for later
    losses = []
    
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch[0].to(device), batch[1].long().to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            preds = outputs.argmax(dim = 1, keepdim = True)
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

    accuracy = running_corrects / len(data_loader.dataset)
    losses = np.average(losses)
    
    print(f'Accuracy is {running_corrects}/{len(data_loader.dataset)} = {accuracy*100}%')
    
    return accuracy, losses

def plot_train_val_losses(t_losses, v_losses, epochs, x_label, y_label, l_label1, l_label2):
    """
    t_losses: your training losses
    v_losses: your validation losses
    epochs: number of epochs trained
    
    x_label: your x-axis name (epochs)
    y_label: your y-axis name (loss)
    l_label1: training loss label name
    l_label2: validation loss label name
    """
    
    # Plot
    plt.figure(figsize=(16,6))
    num_epochs = range(epochs)
    
    plt.xticks(num_epochs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.plot(num_epochs, t_losses, label=l_label1)
    plt.plot(num_epochs, v_losses, label=l_label2)
    plt.legend(loc='upper right')
    plt.show()

def plot_val_accuracies(v_accuracies, epochs, x_label, y_label, a_label1):
    """
    v_accuracies: your validation accuracies
    epochs: number of epochs trained
    
    x_label: your x-axis name (epochs)
    y_label: your y-axis name (accuracy)
    a_label1: validation accuracy label name
    """
    
    # Plot
    plt.figure(figsize=(16,6))
    num_epochs = range(epochs)
    
    plt.xticks(num_epochs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.plot(num_epochs, v_accuracies, label=a_label1)
    plt.legend(loc='upper right')
    plt.show()

def prepare_dataloader(txtfile_path, i_root, crop_size, bs):
    """
    txtfile_path: path leading to the txt file that contains your instances and labels
    i_root: root path of the images
    crop_size: the crop size you want for your image
    bs: batch size you want for your dataloader
    """
    # Create normal transform
    transform = Compose([CenterCrop(crop_size), ToTensor()])
    # Get your data in the form of a list
    data = pd.read_csv(txtfile_path, header=None, sep=" ").values.tolist()
    # Create the dataset using the FlowersDataset class
    dataset = FlowersDataset(i_root, data, crop_size, transform)
    # Lastly create your dataloader
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    
    return dataloader

def train_and_validate(model, epochs, tr_loader, v_loader, criterion, device, optimizer):
    """
    model: your model to train
    epochs: number of iterations to train
    tr_loader: train dataloader
    v_loader: validation dataloader
    criterion: your loss function
    device: what device you are running code on
    optimizer: your weights' updater
    
    """
    # We will use these to determine which epoch had the overall best accuracy, and reflect it later on
    b_accuracy = 0
    b_epoch = 0
    
    # Used to store our training and validation losses for plotting later on
    training_losses = []
    val_losses = []
    
    # Store validation accuracies
    val_accuracies = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs-1}')
        
        # Train and store loss
        train_loss = train(model, tr_loader, criterion, device, optimizer)
        training_losses.append(train_loss)
        # Validate and store loss
        accuracy, val_loss = evaluate(model, v_loader, criterion, device)
        val_accuracies.append(accuracy)
        val_losses.append(val_loss)

        if accuracy > b_accuracy:
            b_weights = model.state_dict()
            b_accuracy = accuracy
            b_epoch = epoch
            print(f'Current best accuracy is {b_accuracy} at epoch {epoch}')
            
    return b_weights, b_epoch, b_accuracy, training_losses, val_losses, val_accuracies

# ## Freeze and Unfreeze layers

# This function freezes all the model parameters
def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

# We then follow up by unfreezing the layers we want to train
def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True

def run():
    ##### Feel free to alter this portion of the code! #####
    
    # Define some parameters to be used
    lr = 0.01 # learning rate
    epochs = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set up your root paths here if needed!
    img_root = 'data/flowers102/102flowers/flowers_data/jpg/'
    traintxt_root = 'data/flowers102/trainfile.txt'
    valtxt_root = 'data/flowers102/valfile.txt'
    testtxt_root = 'data/flowers102/testfile.txt'
    
    
    
    
    # Create datalaoders
    train_loader = prepare_dataloader(traintxt_root, img_root, 224, 16)
    val_loader = prepare_dataloader(valtxt_root, img_root, 224, 16)
    test_loader = prepare_dataloader(testtxt_root, img_root, 224, 16)

    # Set up model (in this case we do mode A where we do not pretrain weights)
    # num_classes=102 reinitializes the resnet last layer to output 102 instead of 1000 classes
    model = models.resnet18(pretrained=True)
    
    # Freeze all layers
    freeze_all(model)
    
    # Unfreeze the layers: layer4 and fc
    unfreeze_layer(model.layer4)
    unfreeze_layer(model.fc)
    
    model.fc = nn.Linear(512, 102)
    model.to(device) 

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.0, weight_decay = 0)

    # Loss Criterion
    criterion = CrossEntropyLoss()
    
    # Run the training and validation phase
    best_weights,best_epoch,best_accuracy,train_losses,val_losses,val_accuracies = train_and_validate(model,epochs,train_loader,val_loader,criterion,device,optimizer)
    
    # Plot out validation accuracies against epoch graph
    plot_val_accuracies(val_accuracies, epochs, 'epochs', 'accuracies', 'Validation Accuracy')
    
    # Plot out the graph for training and validation losses over the epochs
    plot_train_val_losses(train_losses, val_losses, epochs,'epochs','losses','Training Loss', 'Validation Loss')
    
    # We load the best weights we got so far, and evaluate against test set
    model.load_state_dict(best_weights)
    
    print(f'Using the best weights we got from Epoch {best_epoch} which had a validation accuracy of {best_accuracy}')
    print('Testing against the test set we get:')
    evaluate(model, test_loader, criterion, device)

if __name__ == '__main__':
    run()


