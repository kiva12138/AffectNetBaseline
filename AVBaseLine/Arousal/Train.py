import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from Dataset import ArousalDataSet, DataPrefetcher
from AlexNet import AlexNet
import Config
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alex_net = AlexNet()
alex_net.to(device)


image_transforms = transforms.Compose([
    transforms.Resize((Config.image_resize_height, Config.image_resize_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataSetTrain = ArousalDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_train_path)
dataSetValid = ArousalDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_validation_path)

def train_step(net, optimizer, criterion, images, labels):
    optimizer.zero_grad()
    outputs = net(images).type(dtype=torch.double)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def valid_step(net, criterion, valid_images, valid_labels):
    valid_outputs = net(valid_images)
    loss = criterion(valid_outputs, valid_labels)
    return loss.item()

class ConcordanceCorCoeff(nn.Module):
    def __init__(self):
        super(ConcordanceCorCoeff, self).__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std
    def forward(self, prediction, ground_truth):
        mean_gt = self.mean (ground_truth, 0)
        mean_pred = self.mean (prediction, 0)
        var_gt = self.var (ground_truth, 0)
        var_pred = self.var (prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum (v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2)) * self.sqrt(self.sum(v_gt ** 2)))
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator=2*cor*sd_gt*sd_pred
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/denominator
        return 1-ccc

if __name__ == "__main__":

    # Data Preparing
    torch.set_default_dtype(torch.float64)
    traindataloader = DataLoader(dataSetTrain, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    validdataloader = DataLoader(dataSetValid, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    train_length = dataSetTrain.length
    valid_length = dataSetValid.length
    train_iter_per_epoch = len(traindataloader)
    valid_iter_per_epoch = len(validdataloader)
    print(train_length, valid_length, train_iter_per_epoch, valid_iter_per_epoch)
    print('Data prepare ready ====== ')

    optimizer = optim.SGD(alex_net.parameters(), lr=Config.lr, momentum=Config.momentum)
    criterion = nn.MSELoss()

    writer = SummaryWriter(Config.log_dir)
    early_stop = False
    early_stop_record = [1000.0, 500.0]

    print('Start Training ====== ')
    for epoch in range(Config.EPOCHS):

        # Training Step
        train_prefetcher = DataPrefetcher(traindataloader)
        train_data = train_prefetcher.next()
        iteration  = 0
        running_loss = 0.0

        while (train_data is not None) and (not early_stop):
            iteration  += 1 

            # Get Data
            images, arousals = train_data['image'], train_data['arousal'].view(-1, 1)
            if not (images.is_cuda and arousals.is_cuda):
                images, arousals = images.to(device), arousals.to(device)

            # Train Step
            running_loss += train_step(alex_net, optimizer, criterion, images, arousals)

            # Valid And Log
            if iteration % Config.check_iter == 0 or train_iter_per_epoch==iteration:

                # Get Validation Loss
                running_loss_computed = 0.0
                if train_iter_per_epoch==iteration:
                    running_loss_computed = running_loss / (train_iter_per_epoch % Config.check_iter)
                else:
                    running_loss_computed = running_loss / Config.check_iter

                # Scalar And Log
                print('[%d, %5d] loss: %.3f' % (epoch + 1, iteration, running_loss_computed))
                writer.add_scalar('training loss',  running_loss_computed,             epoch*train_iter_per_epoch+iteration)
                writer.add_scalar('training rmse',  math.sqrt(running_loss_computed),  epoch*train_iter_per_epoch+iteration)
                running_loss = 0.0
                
                # Validation Step
                with torch.no_grad():
                    valid_prefetcher = DataPrefetcher(validdataloader)
                    valid_data = valid_prefetcher.next()
                    valid_iteration  = 0
                    valid_loss = 0.0
                    valid_loss_sum = 0.0  # Loss Of All Validation Data
                    while valid_data is not None:
                        valid_iteration  += 1 

                        # Get Data
                        valid_images, valid_arousals = valid_data['image'], valid_data['arousal'].view(-1, 1)
                        if not (valid_images.is_cuda and valid_arousals.is_cuda):
                            valid_images, valid_arousals = valid_images.to(device), valid_arousals.to(device)

                        # Valid Step
                        valid_loss += valid_step(alex_net, criterion, valid_images, valid_arousals)

                        # Log Valid Step
                        if valid_iteration % Config.check_iter == 0 or valid_iter_per_epoch==valid_iteration:

                            # Get Validation Loss
                            valid_loss_computed = 0.0
                            if valid_iter_per_epoch==valid_iteration:
                                valid_loss_computed = valid_loss / (valid_iter_per_epoch % Config.check_iter)
                            else:
                                valid_loss_computed = valid_loss / Config.check_iter
                            
                            # Scalar And Log
                            print('Valid [%d,  %d] loss: %.3f' % (epoch + 1, valid_iteration, valid_loss_computed))
                            writer.add_scalar('valid loss', valid_loss_computed,             epoch*train_iter_per_epoch + iteration)
                            writer.add_scalar('valid rmse', math.sqrt(valid_loss_computed),  epoch*train_iter_per_epoch + iteration)

                            valid_loss_sum += valid_loss_computed
                            valid_loss = 0.0
                        
                        valid_data = valid_prefetcher.next()
                    

                    print('Valid iter[%d] loss: %.3f' % (epoch*train_iter_per_epoch, valid_loss_sum / valid_iter_per_epoch))

                    # Record Recent Validation Loss
                    if len(early_stop_record) < Config.patience:
                        early_stop_record.append(valid_loss_sum / valid_iter_per_epoch)
                    else:
                        early_stop_record = early_stop_record[1:]
                        early_stop_record.append(valid_loss_sum / valid_iter_per_epoch)
        
                    del valid_prefetcher
            
                # Check whether Overfit
                early_stop = True
                for loss_i in range(0, len(early_stop_record)-1):
                    if early_stop_record[loss_i] > early_stop_record[len(early_stop_record)-1]:
                        early_stop = False
                        break
            
            train_data = train_prefetcher.next()
        
        if early_stop:
            print("Early Stop ====== ")
            break

        del train_prefetcher

    print('Training Over ====== ')
    writer.close()
    torch.save(alex_net, Config.weights_save_path)
    print('Model Saved to {} ====== '.format(Config.weights_save_path))
