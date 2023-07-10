#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from attacks import PGD, FGSM


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))

# class ConvBNRelu(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, x):
#         x = self.relu(self.bn(self.conv(x)))
#         return x
    
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.convBlock1 = ConvBNRelu(channels, channels)
#         self.convBlock2 = ConvBNRelu(channels, channels)

#     def forward(self, x):
#         out = self.convBlock1(x)
#         out = self.convBlock2(x)
#         out = out + x
#         return out

# class Net(nn.Module):
#     """Essentially the ResNet9 architecture"""

#     model_file="models/default_model.pth"
#     '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

#     def __init__(self):
#         super().__init__()
#         self.conv1 = ConvBNRelu(3, 64)
#         self.conv2 = ConvBNRelu(64, 128)
#         self.res1 = ResBlock(128)
        
#         self.conv3 = ConvBNRelu(128, 256)
#         self.conv4 = ConvBNRelu(256, 512)
#         self.res2 = ResBlock(512)
        
#         self.pool = nn.MaxPool2d(2, 2)
        
#         self.final_pool = nn.MaxPool2d(4, 4)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512, 10)
        
#     def forward(self, x):
#         x = self.pool(self.conv2(self.conv1(x)))
#         x = self.res1(x)
#         x = self.pool(self.conv3(x))
#         x = self.pool(self.conv4(x))
#         x = self.res2(x)
#         x = self.fc(self.flatten(self.final_pool(x)))
#         x = F.log_softmax(x, dim=1)
#         return x



def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[0].to(device) + torch.randn_like(data[0], device='cuda') * 0.45 , data[1].to(device)
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def train_model_adversarial(net, train_loader, pth_filename, num_epochs, l2=False):
    '''Training function for adverserial training'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[0].to(device) + torch.randn_like(data[0], device='cuda') * 0.45 , data[1].to(device)
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # avoid wasting time on useless grads
            net.requires_grad = False
            # get attacked images
            if l2:
                attacked_images = PGD(net, inputs, labels, criterion, epsilon=0.045, delta=0.02, l2=True)
            else:
                attacked_images = PGD(net, inputs, labels, criterion, epsilon=0.031, l2=False)
            
            # reactivate gradient computation
            net.requires_grad = True
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Both the original image and the attacked version must be correctly labelled
            outputs = net(inputs)
            attacked_outputs = net(attacked_images)
            loss = (criterion(outputs, labels) + criterion(attacked_outputs, labels))/2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def train_model_mix_adversarial(net, train_loader, pth_filename, num_epochs):
    '''Training function for mix adverserial training'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[0].to(device) + torch.randn_like(data[0], device='cuda') * 0.45 , data[1].to(device)
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # avoid wasting time on useless grads
            net.requires_grad = False
            # get attacked images
            attacked_l2_images = PGD(net, inputs, labels, criterion, epsilon=0.045, delta=0.02, l2=True)
            attacked_linf_images = PGD(net, inputs, labels, criterion, epsilon=0.031, l2=False)
            
            # reactivate gradient computation
            net.requires_grad = True
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Both the original image and the attacked versions must be correctly labelled
            outputs = net(inputs)
            attacked_l2_outputs = net(attacked_l2_images)
            attacked_linf_outputs = net(attacked_linf_images)
            loss = (criterion(outputs, labels) + criterion(attacked_l2_outputs, labels) + criterion(attacked_linf_outputs, labels))/3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_FGSM(net, test_loader):
    '''FGSM Testing function.'''
    criterion = nn.NLLLoss()
    correct = 0
    total = 0
    # we're not training, but we still need to calculate the gradients
    for i,data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        # generate the attack labels using FGSM
        attacked_images = FGSM(net, images, labels, criterion)
        
        # calculate outputs by running the attecked images through the network
        outputs = net(attacked_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_PGD_linf(net, test_loader):
    '''PGD-linf Testing function.'''
    criterion = nn.NLLLoss()
    correct = 0
    total = 0
    # we're not training, but we still need to calculate the gradients
    for i,data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        # generate the attack labels using PGD
        attacked_images = PGD(net, images, labels, criterion,epsilon=0.031, delta=0.02, l2=False)
        
        # calculate outputs by running the attecked images through the network
        outputs = net(attacked_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_PGD_l2(net, test_loader):
    '''PGD-l2 Testing function.'''
    criterion = nn.NLLLoss()
    correct = 0
    total = 0
    # we're not training, but we still need to calculate the gradients
    for i,data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        # generate the attack labels using PGD
        attacked_images = PGD(net, images, labels, criterion,epsilon=0.045, delta=0.03, l2=True)
        
        # calculate outputs by running the attecked images through the network
        outputs = net(attacked_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total



def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")

    parser.add_argument('-a', '--adversarial', action="store_true",
                        help="Used adversarial training during training.")
    
    parser.add_argument('--l2', action="store_true",
                        help="if --adversarial is set, use l2 instead of linf.")
    parser.add_argument('--mix', action="store_true",
                        help="if --adversarial is set, use mix-adversarial.")
    
    
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        if args.adversarial:
            if args.mix:
                print(f"Using mix-adversarial training")
                train_model_mix_adversarial(net, train_loader, args.model_file, args.num_epochs)
            else:
                print(f"Using adversarial training with {'l2' if args.l2 else 'linf'} PGD")
                train_model_adversarial(net, train_loader, args.model_file, args.num_epochs, l2=args.l2)
            
        else:
            train_model(net, train_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print(f"Model natural accuracy (valid): {acc}")
    
    
    acc = test_FGSM(net, valid_loader)
    print(f"Model FGSM accuracy (valid): {acc}")
    
    
    acc = test_PGD_linf(net, valid_loader)
    print(f"Model PGD-linf accuracy (valid): {acc}")
    
    acc = test_PGD_l2(net, valid_loader)
    print(f"Model PGD-l2 accuracy (valid): {acc}")

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

