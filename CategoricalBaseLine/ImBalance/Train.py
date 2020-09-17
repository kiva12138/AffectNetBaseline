import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import Config
from AlexNet import AlexNet
from Dataset import DataPrefetcher, ExpressionDataSet
from Utils import MeanManager, AccuracyManager


def train_step(net, traindataloader, train_iter_per_epoch, optimizer, criterion, writer, lr_scheduler, epoch):
    
    train_prefetcher = DataPrefetcher(traindataloader)
    train_data = train_prefetcher.next()

    iteration  = 0

    running_loss_manager = MeanManager()

    while train_data is not None:
        iteration  += 1
        
        # Get Data
        images, expressions = train_data['image'], train_data['expression']
        if not (images.is_cuda and expressions.is_cuda):
            images, expressions = images.to(device), expressions.to(device)

        # Train
        optimizer.zero_grad()
        outputs = net(images).type(dtype=torch.double)
        loss = criterion(outputs, expressions)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        running_loss_manager.update(val=loss.item(), n=expressions.size(0))

        # Log
        if iteration % Config.check_iter == 0 or train_iter_per_epoch==iteration:
            print('[%d, %5d] loss: %.3f, lr: %.9f' % (epoch+1, iteration, running_loss_manager.mean, lr_scheduler.get_last_lr()[-1]))
            writer.add_scalar('training loss',  running_loss_manager.mean, epoch*train_iter_per_epoch+iteration)
            writer.add_scalar('lr',  lr_scheduler.get_last_lr()[-1], epoch*train_iter_per_epoch+iteration)
            running_loss_manager.reset()
            
        train_data = train_prefetcher.next()

    del train_prefetcher


def valid_step(net, validdataloader, criterion, writer, epoch):

    with torch.no_grad():
        valid_prefetcher = DataPrefetcher(validdataloader)
        valid_data = valid_prefetcher.next()
        iteration = 0

        valid_loss_manager = MeanManager()
        valid_acc_manager = AccuracyManager(Config.num_classes)

        while valid_data is not None:
            iteration  += 1 

            valid_images, valid_expressions = valid_data['image'], valid_data['expression']
            if not (valid_images.is_cuda and valid_expressions.is_cuda):
                valid_images, valid_expressions = valid_images.to(device), valid_expressions.to(device)

            valid_outputs = net(valid_images)
            loss = criterion(valid_outputs, valid_expressions)

            valid_loss_manager.update(val=loss.item(), n=valid_expressions.size(0))

            _, predicted = torch.max(valid_outputs.data, 1)
            c = (predicted == valid_expressions).squeeze()
            for i in range(c.size()[0]):
                valid_acc_manager.update(cls=valid_expressions.squeeze()[i], right=c[i])

            valid_data = valid_prefetcher.next()
        
        print('Valid Epoch[%d] Loss: %.3f, MA: %.4f, Accuracy: %.4f' % (epoch, valid_loss_manager.mean, valid_acc_manager.ma, valid_acc_manager.accuracy))
        writer.add_scalar('Valid MA', valid_acc_manager.ma, epoch)
        writer.add_scalar('Valid ACC', valid_acc_manager.accuracy, epoch)
        writer.add_scalar('Valid Loss', valid_loss_manager.mean, epoch)
        for exp_i in range(Config.num_classes):
            print('Accuracy of %5s : %2d %%' % ( [exp_i], 100*valid_acc_manager.class_accuracy[exp_i]))
            writer.add_scalar('acc'+str(exp_i), 100*valid_acc_manager.class_accuracy[exp_i], epoch)
        valid_loss_manager.reset()
        valid_acc_manager.reset()

        del valid_prefetcher
    
    torch.save(net, Config.weights_save_path)
    print('Model saved to ' + Config.weights_save_path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Preparing data......')
    image_transforms = transforms.Compose([
        transforms.Resize((Config.image_resize_height, Config.image_resize_width)),
        transforms.RandomCrop(size=(Config.image_crop_height, Config.image_crop_height)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_transforms_valid = transforms.Compose([
        transforms.Resize((Config.image_crop_height, Config.image_crop_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataSetTrain = ExpressionDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_train_path)
    dataSetValid = ExpressionDataSet(transform=image_transforms_valid, path=Config.manually_annotated_file_list_validation_path)

    # Data Preparing
    traindataloader = DataLoader(dataSetTrain, batch_size=Config.BATCH_SIZE, shuffle=True)
    validdataloader = DataLoader(dataSetValid, batch_size=Config.BATCH_SIZE, shuffle=False)

    train_length = dataSetTrain.length
    valid_length = dataSetValid.length
    train_iter_per_epoch = len(traindataloader)
    valid_iter_per_epoch = len(validdataloader)


    print("Train data length = ", train_length)
    print("Valid data length = ", valid_length)
    print("Train iteration per epoch = ", train_iter_per_epoch)
    print("Valid iteration per epoch = ", valid_iter_per_epoch)

    print("Preparing net......")
    alex_net = None
    if Config.continue_train:
        print('Loading previous net......')
        alex_net = torch.load(Config.weights_save_path)
    else:
        print('Creating new net.....')
        alex_net = AlexNet(num_classes=Config.num_classes)
    alex_net.to(device)
    optimizer = optim.SGD(alex_net.parameters(), lr=Config.lr, momentum=Config.momentum)
    lr_schedule = optim.lr_scheduler.StepLR(optimizer, Config.lr_decrease_iter, Config.lr_decrease_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(Config.log_dir)

    print('Start Training......')
    for epoch in range(Config.EPOCHS): 
        if Config.continue_train and epoch <= Config.continue_epoch:
            for i in range(train_iter_per_epoch):
                lr_schedule.step()
            continue
        train_step(alex_net, traindataloader, train_iter_per_epoch, optimizer, criterion, writer, lr_schedule, epoch)
        valid_step(alex_net, validdataloader, criterion, writer, epoch)
        
    writer.close()
    print('Over Training......')
