import numpy as np, os, sys

import matplotlib.pyplot as plt #patch-wise similarities, droi images
from matplotlib import ticker, cm

import torch.nn as nn
import torch.utils.data 

from torchvision import datasets, transforms
import torch.optim

class FashionNet(nn.Module):
    def __init__(self, batch_size, val_size, tr_size):
        # linear1, linear2 and linear3 follow l1, l2, and l3 respectively
        
        super(FashionNet, self).__init__()
        self.linear1 = nn.Linear(28*28, 300) # l1 -> input of 28 * 28 follows the tensor shape of FashionMNIST
        self.linear2 = nn.Linear(300, 100) # l2
        self.linear3 = nn.Linear(100, 10) # l3
        self.batch_size = batch_size
        self.val_size = val_size
        self.tr_size = tr_size
        
    def forward(self, x):
        
        # Need to reshape as the input is 2800 x 28, so we reshape into 100 x 28 x 28
        x = torch.reshape(x,(-1,28*28))
        
        # First layer and first relu
        first_out = self.linear1(x)
        first_relu_out = torch.relu_(first_out)
        
        # Second hidden layer
        second_out = self.linear2(first_relu_out)
        second_relu_out = torch.relu_(second_out)
        
        # Y predictions
        y_pred = self.linear3(second_relu_out)
        
        return y_pred

def train_epoch(lr, model, trainloader, criterion, device, optimizer):
    model.train()
 
    losses = list()
    for batch_idx, data in enumerate(trainloader):

        inputs=data[0].to(device)
        labels=data[1].to(device)
        optimizer.zero_grad()
        
        output = model(inputs)
        
        loss = criterion(output, labels)
        loss.backward()
        
        optimizer.step()
        losses.append(loss.item())

    return losses

def evaluate(model, dataloader, criterion, device, is_test=False, is_train=False):

    model.eval()

    gtpos=0
    gtneg=0
    tps=0
    tns=0
    fbeta=1

    running_corrects = 0
    
    # Check if using traing, val or test datasets
    if is_test:
        data_size = len(dataloader.dataset)
    elif is_train:
        data_size = model.tr_size
    else:
        data_size = model.val_size
    
    # Store val or test losses
    val_test_losses = list()
    
    with torch.no_grad():
        for ctr, data in enumerate(dataloader):

            print ('epoch at',len(dataloader.dataset), ctr)
            inputs = data[0].to(device)        
            outputs = model(inputs)

            labels = data[1]
            
            # Calculate val/test loss
            val_test_loss = criterion(outputs, labels)
            val_test_losses.append(val_test_loss.item())
            
            labels = labels.float()
            cpuout= outputs.to('cpu')

            _, preds = torch.max(cpuout, 1)


            running_corrects += torch.sum(preds == labels.data)            
            accuracy = running_corrects.double() / data_size 

    return accuracy.item(), val_test_losses

def train_modelcv(lr, dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device):

    best_measure = 0
    best_epoch =-1

    # Stores the averaged loss over all minibatches for each epoch
    tr_averaged_loss = []
    val_tes_averaged_loss = []
    val_tes_accuracies = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train(True)
        losses=train_epoch(lr, model,  dataloader_cvtrain,  criterion,  device , optimizer)
        
        # Append to training average loss
        tr_averaged_loss.append(np.average(losses))
        
        #scheduler.step()

        model.train(False)
        measure, val_test_losses = evaluate(model, dataloader_cvtest, criterion, device)
        print(' perfmeasure', measure)
        
        # Append to val_accuracies
        val_tes_accuracies.append(measure)
        
        # Append to val_test average loss
        val_tes_averaged_loss.append(np.average(val_test_losses))
        
        # Store the weight
        torch.save(model.state_dict(), f'epoch_checkpoints/epoch_{epoch}_model.ckpt')
        
        if measure > best_measure: #higher is better or lower is better?
            bestweights= model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', measure, ' at epoch ', best_epoch)

    return best_epoch, best_measure, bestweights, tr_averaged_loss, val_tes_averaged_loss, val_tes_accuracies

def plot_it(tr_acc, val_acc, test_acc, tr_loss, val_loss, test_loss):
    
    plt.figure(figsize=(20,5))
    plt.subplot(121),
    plt.plot(tr_acc, label = "Training Accuracy"),
    plt.plot(val_acc, label = "Validation Accuracy"),
    plt.plot(test_acc, label = "Test Accuracy"),
    plt.title('Accuracy')
    plt.subplot(122),
    plt.plot(tr_loss, label = "Training Loss"),
    plt.plot(val_loss, label = "Validation Loss"),
    plt.plot(test_loss, label = "Test Loss"),
    plt.title('Loss')
    plt.legend(frameon = False)
    plt.show()

def run():
    
    maxnumepochs = 5
    validation_size = 0.1
    batch_size = 100
    learning_rate = 0.01

    torch.manual_seed(1) # Set random seed

    # Fetch and download FashionMNIST dataset from torchvision and transform to TensorDataset
    tr_data = datasets.FashionMNIST("FashionMNIST_data", train = True, download = True, transform = transforms.ToTensor())
    tes_data = datasets.FashionMNIST("FashionMNIST_data", train = False, download = True, transform = transforms.ToTensor())


    ### Create a sampler for training and validation dataloaders ###

    # Prepare the index ranges for splitting our train and val first
    num_instances = len(tr_data)
    indexes = list(range(num_instances)) # a list of indexes for the samplers
    t_v_split = int(np.floor(validation_size * num_instances)) # use this to split up our train and val

    tr_indexes = indexes[t_v_split:]
    val_indexes = indexes[:t_v_split]

    # Create the samplers
    tr_sampler = torch.utils.data.sampler.SubsetRandomSampler(tr_indexes)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indexes)

    # Create our dataloaders for training and validation using our samplers
    trloader = torch.utils.data.DataLoader(tr_data, batch_size = batch_size, sampler = tr_sampler)
    valloader = torch.utils.data.DataLoader(tr_data, batch_size = batch_size, sampler = val_sampler)


    # Initialize Model
    model = FashionNet(batch_size, t_v_split, num_instances)

    # Initialize Loss Function, in this case we use cross entropy for 10 classes 
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer SGD
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.0, weight_decay = 0)

    # Set up device
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device('cpu')


    best_epoch, best_perfmeasure, bestweights, train_averaged_loss, val_averaged_loss, val_accuracies = train_modelcv(lr = learning_rate, dataloader_cvtrain = trloader, dataloader_cvtest = valloader ,  model = model ,  criterion = criterion , optimizer = optimizer, scheduler = None, num_epochs = maxnumepochs , device = device)

    ## Create test dataloader
    tesloader = torch.utils.data.DataLoader(tes_data, batch_size = 100, shuffle = False)

    # We loop through the different weights from each epoch to get the test losses
    test_averaged_loss = []
    test_accuracies = []

    train_accuracies = []

    for i in range(maxnumepochs):
        # Load the weights
        model.load_state_dict(torch.load(f'epoch_checkpoints/epoch_{i}_model.ckpt'))

        # Test
        test_accuracy, test_loss = evaluate(model = model, dataloader = tesloader, criterion = criterion, device = device, is_test = True)

        test_averaged_loss.append(np.average(test_loss))
        test_accuracies.append(test_accuracy)

        # Train
        train_accuracy, _ = evaluate(model = model, dataloader = trloader, criterion = criterion, device = device, is_train = True)

        train_accuracies.append(train_accuracy)

        
        
    plot_it(train_accuracies, val_accuracies, test_accuracies, train_averaged_loss, val_averaged_loss, test_averaged_loss)

if __name__=='__main__':

  run()
