# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:26:48 2023

@author: Eduin Hernandez
"""
#-----------------------------------------------------------------------------
'Libraries'
#Time
import time
from datetime import datetime
from tqdm import tqdm

#Storage
import argparse
from utils.parser_utils import str2bool

#Math
import numpy as np

#DNN
import torch
import torch.nn as nn
from torch import optim
from models.sample_models import models_dict as sample_models
import torchvision.datasets as datasets
from torchvision import transforms

#Plotting
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for CIFAR10/100 Training')

    'Model Details'
    parser.add_argument('--model-name', type=str, default='VGG', help='Model used for the training.')
    parser.add_argument('--batch-size', type=int, default=120, help='Batch Size for Training')
    parser.add_argument('--epoch-num', type=int, default=10, help='Total Epochs used for Training')
    parser.add_argument('--layers', type=int, default=16, help='Number of Layers in Student Network')
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning Rate for the model')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight Decay for model')

    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    parser.add_argument('--dampening', type=float, default=0, help='Dampening for model')
    parser.add_argument('--nesterov', type=str2bool, default='False', help='Whether to use Nesterov Momentum')
    
    'Dataset'
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to Use. Can be CIFAR10 or CIFAR100')
    parser.add_argument('--dataset-path', type=str, default='D:/Dewen/datasets/', help='Path that contains the dataset. Avoids redownloading.')
    
    'Device'
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Either cpu or cuda for gpu')
    
    'Simulations'
    parser.add_argument('--simulation-num', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--use-manual-seed', type=str2bool, default='True', help='Whether to use manual seed or random seed.')
    
    'Print and Plot'
    parser.add_argument('--plot-state', type=str2bool, default='True', help='Whether to plot the results')
    parser.add_argument('--verbose-state-sim', type=str2bool, default='True', help='Whether to print the results per simulation')
    parser.add_argument('--verbose-state', type=str2bool, default='True', help='Whether to print the results at end of code')
    
    args = parser.parse_args()
    return args
            
#------------------------------------------------------------------------------
def run_experiment(args):
    'Preparing and Checking Parser'
    if args.optimizer not in ['SGD', 'Adam', 'RAdam', 'RMSprop']:
        assert False, 'Optimizer not found'
    else:
        if args.optimizer == 'SGD':
            func_optim = optim.SGD
        elif args.optimizer == 'Adam':
            func_optim = optim.Adam
        elif args.optimizer == 'RAdam':
            func_optim = optim.RAdam
        elif args.optimizer == 'RMSprop':
            func_optim = optim.RMSprop
    
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if args.use_manual_seed:
        torch.manual_seed(0)
    
    assert args.model_name in list(sample_models.keys()), 'Invalid Training Model!'

    #-----------------------------------------------------------------------------
    'Data Preparation'
    transformTrain = transforms.Compose( [transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    transformTest = transforms.Compose( [transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    if args.dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transformTrain) 
        dataset_test = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transformTest)
    elif args.dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transformTrain) 
        dataset_test = datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transformTest)
    else:
        assert False, 'Selected the wrong dataset'
    
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                             shuffle=False)
    
    train_size = len(dataset_train)
    test_size = len(dataset_test)
    #------------------------------------------------------------------------------
    'Training'
    if args.verbose_state:
        print('Device:', device.type)
        momentum = ''
        layers = ''
        if args.optimizer == 'SGD':
            momentum = ', Moment: '+ str(args.momentum)
        if args.model_name == 'VGG' or args.model_name == 'ResNet':
            layers = '-' + str(args.layers)
        
        print(args.dataset + ' - ' + args.model_name + layers + ' - ' + args.optimizer + ': ' + str(args.learning_rate) + momentum + ' - Start Time: ' + datetime.now().strftime("%H:%M:%S"))
    
    simulation_start_time = time.time()
    
    losses = np.zeros((args.simulation_num, args.epoch_num))
    losses_val = np.zeros((args.simulation_num, args.epoch_num))
    
    acc = np.zeros((args.simulation_num, args.epoch_num))
    acc_val = np.zeros((args.simulation_num, args.epoch_num))
    
    speed = np.zeros(args.simulation_num)

    sim_bar = tqdm(total=args.simulation_num, position=0, disable=not(args.verbose_state))
    for sim_num in range(args.simulation_num):
        if args.use_manual_seed:
            torch.manual_seed(sim_num + 1)
            
        'Model Initialization'
        net = sample_models[args.model_name](args.layers, 10).to(device)
        
        'Loss and Optimizer'
        criterion = nn.CrossEntropyLoss()
        criterion_sum = nn.CrossEntropyLoss(reduction='sum')
        
        if args.optimizer == "SGD":
            optimizer = func_optim(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                    momentum=args.momentum, dampening=args.dampening, nesterov=args.nesterov)
        else:
            optimizer = func_optim(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            
        start_time = time.time()
        
        pbar = tqdm(total=args.epoch_num, position=1, disable=not(args.verbose_state_sim))
        'Train and Validate Network'
        for epoch in range(args.epoch_num):  # loop over the dataset multiple times
            correct = 0
            correct_val = 0
            running_loss = 0.0
            running_loss_val = 0.0
                
            'Training Phase'
            net.train()
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()
        
                'Foward Propagation'
                output = net(inputs)
                loss = criterion(output, labels)
                
                'Backward Propagation'
                #Automatically calculates the gradients for trainable weights, access as weight.grad
                loss.backward()
                
                #Performs the weight Update
                optimizer.step()
        
                #statistics
                running_loss += criterion_sum(output, labels).item()
                predicted = output.argmax(dim=1)
                correct += (predicted == labels).sum().item()
            
            net.eval()
            with torch.no_grad():
                for input_val, labels_val in testloader:
                    input_val, labels_val = input_val.to(device), labels_val.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    'Foward Propagation'
                    output = net(input_val)
                    loss = criterion(output, labels_val)
                    
                    #statistics
                    running_loss_val += criterion_sum(output, labels_val).item()
                    predicted = output.argmax(dim=1)
                    correct_val += (predicted == labels_val).sum().item()
                
            #Statistics   
            acc[sim_num, epoch] = correct/ train_size
            losses[sim_num, epoch] = running_loss / train_size
            acc_val[sim_num, epoch] = correct_val/ test_size
            losses_val[sim_num, epoch] = running_loss_val / test_size
    
            pbar.update(1)

        speed[sim_num] = time.time() - start_time
        pbar.close()
        
        sim_bar.set_description('Model: %d, Acc: %.5f, Loss: %.5f, Val_Acc: %.5f, Val_Loss: %.5f, Time: %d s' % (sim_num + 1,
                acc[sim_num,-1], losses[sim_num,-1], acc_val[sim_num,-1], losses_val[sim_num,-1], speed[sim_num]),
                refresh=False)
        
        sim_bar.update(1)
            
    sim_bar.close()
    
    #------------------------------------------------------------------------------
    'Plotting Results'
    if args.plot_state:
        plt.figure()
        plt.title('Loss - ' + args.code_type)
        plt.plot(losses.mean(axis=0), color='C2', label='Train Loss')
        plt.plot(losses_val.mean(axis=0), color='C3',label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.legend()
        plt.grid()    
        
        plt.figure()
        plt.title('Acc - ' + args.code_type)
        plt.plot(acc.mean(axis=0), color='C2', label='Train Acc')
        plt.plot(acc_val.mean(axis=0), color='C3',label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.grid() 
    #------------------------------------------------------------------------------
    'Printing Overall Statistics'
    if args.verbose_state:
        print('\nModels: %d, Avg. Acc: %.5f, Avg. Loss: %.5f, Avg. Val Acc: %.5f, Avg. Val Loss: %.5f, Avg Time: %d s' % (args.simulation_num,
                    acc[:,-1].mean(), losses[:,-1].mean(), acc_val[:,-1].mean(), losses_val[:,-1].mean(), speed.mean()))
        
        momentum = ''
        layers = ''
        if args.optimizer == 'SGD':
            momentum = ', Moment: '+ str(args.momentum)
        if args.model_name == 'VGG' or args.model_name == 'ResNet':
                layers = '-' + str(args.layers)
                
        print(args.dataset + ' - ' + args.model_name + layers + ' - ' + args.optimizer + ': ' + str(args.learning_rate) + momentum + ' -  End Time: ' + datetime.now().strftime("%H:%M:%S"))
        
    end_time = time.time() - simulation_start_time
    print('Elapsed Time: %.2dD:%.2dH:%.2dM:%.2dS' % (end_time / 86400, (end_time / 3600) % 24, (end_time / 60) % 60, end_time % 60))

if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)