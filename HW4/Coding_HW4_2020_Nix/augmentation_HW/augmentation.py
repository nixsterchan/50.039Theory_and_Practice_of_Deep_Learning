import torch
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from torchvision import models
from torchvision.transforms import FiveCrop, ToTensor, Lambda, Compose, CenterCrop, Normalize

import pandas as pd
import numpy as np
import torch.nn as nn

from data.imagenet import getimagenetclasses as get_labels

from os import listdir
from os.path import isfile, join

# ## Create the Dataset Class
class HandsomeBinderNet(Dataset):
    def __init__(self, img_root, ins_label_pairs , crop_size, transform=None):
        
        """
        
        img_root: contains the path to the image root folder
        ins_label_pairs: instance label pair that contains a list of all the image path names and their respective labels
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
        img = (Image.open(image_path))
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

# Prepares the dataset
def prepare_dataloader(img_path, ins_label_pairs, crop_size, transform, bs):
    """
    img_path: path to image root
    ins_label_pairs: instance label pairs containing the paths of jpeg images and their respective labels
    crop_size: your desired crop size
    transform: your transformation sequence
    bs: your desired batch size for dataloader
    """

    # Create dataset and dataloader
    dataset = HandsomeBinderNet(img_path, ins_label_pairs, crop_size=crop_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    
    return dataloader

# Runs evaluation
def evaluate(dataloader, model, device, is_fivecrop=False):
    """
    dataloader: dataloader
    model: your model
    device: what you are using to compute
    is_fivecrop: default False. if set to True, it deals with the 5 tensor for problem 2
    """
    model.eval()
    # Calculate the accuracy
    num_corrects = 0

    with torch.no_grad():

        if (is_fivecrop):
            for i, batch in enumerate(dataloader):
                images, labels = batch[0].to(device), batch[1].to(device)

                # Images is a five-tensor
                bs, ncrops, c, h, w = images.size()

                # fuse batch size and ncrops
                result = model(images.view(-1, c, h, w)) 

                # avg over crops
                result_avg = result.view(bs, ncrops, -1).mean(1) 

                pred = result_avg.argmax(dim=1, keepdim=True)
                num_corrects += pred.eq(labels.view_as(pred)).sum().item()
        else:
            for i, batch in enumerate(dataloader):
                images, labels = batch[0].to(device), batch[1].to(device)
                output = model(images)
                pred = output.argmax(dim=1, keepdim=True)
                num_corrects += pred.eq(labels.view_as(pred)).sum().item()


    print(f'Accuracy: {100 * num_corrects/len(dataloader.dataset)}%')

def problem1(imgpath, ins_label_pairs, device):
    #### Problem 1 Test Performance of a pretrained net with and without normalizing ####
    print('Problem 1: Test Performance of a pretrained net with and without normalizing \n \n \n \n')
    
    ## Create transformations for normalize and none ##
    # No normalize
    t_no_normalize = Compose([CenterCrop(224), ToTensor()])
    # With normalize
    normalizer = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_normalize = Compose([CenterCrop(224), ToTensor(), normalizer])
    
    # Evaluate without normalize
    dataloader = prepare_dataloader(imgpath, ins_label_pairs, 224, t_no_normalize, 16)
    model = models.resnet18(pretrained=True).to(device)
    print('For Problem 1, in the case without normalization:')
    evaluate(dataloader, model, device)
    
    # Evaluate with normalize
    dataloader = prepare_dataloader(imgpath, ins_label_pairs, 224, t_normalize, 16)
    model = models.resnet18(pretrained=True).to(device)
    print('For Problem 1, in the case without normalization:')
    evaluate(dataloader, model, device)

def problem2(imgpath, ins_label_pairs, device):
    #### Problem 2 Test the performance of a pretrained net five crop ####
    print('\n \n \n \nProblem 2: Test the performance of a pretrained net five crop \n \n \n \n')

    # Create transformations for five crop
    normalizer = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_five_crop = Compose([FiveCrop(224), 
                                      Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                                      Lambda(lambda crops: torch.stack([normalizer(crop) for crop in crops]))])
    # Evaluate
    dataloader = prepare_dataloader(imgpath, ins_label_pairs, 280, t_five_crop, 16)
    model = models.resnet18(pretrained=True).to(device)
    print('For Problem 2, with five crop:')
    evaluate(dataloader, model, device, is_fivecrop=True)

def problem3(imgpath, ins_label_pairs, device):
    #### Problem 3 Different input size of neural networks with different pretrained neural nets ####
    print('\n \n \n \nProblem 3: Different input size of neural networks with different pretrained neural nets \n \n \n \n')
    
    # Normalize
    normalizer = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_normalize = Compose([CenterCrop(224), ToTensor(), normalizer])
    
    # Create Dataloader
    dataloader = prepare_dataloader(imgpath, ins_label_pairs, 330, t_normalize, 16)

    # Evaluate for DenseNet 161
    model = models.densenet161(pretrained=True).to(device)
    print('For Problem 3, using the Dense Net 161:')
    evaluate(dataloader, model, device)
    
    # Evaluate for GoogleNet
    model = models.googlenet(pretrained=True).to(device)
    print('For Problem 3, using the Google Net:')
    evaluate(dataloader, model, device)

def run():
    # Get path for images and their JPEG path (Change this if needed!)
    imgpath = 'data/imagenet/imagenet2500/imagespart/'

    # Store the JPEG path names into a list
    onlyfiles = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
    
    # Get the JPEG file paths and their respective labels into a instance-label list
    ins_label_pairs = []
    for JPEG in onlyfiles:
        ins_label_pairs.append([JPEG , get_labels.test_parseclasslabel(JPEG)])
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
    # Problem 1
    problem1(imgpath, ins_label_pairs, device)
    # Problem 2
    problem2(imgpath, ins_label_pairs, device)
    # Problem 3
    problem3(imgpath, ins_label_pairs, device)  

if __name__ == '__main__':
    run()


