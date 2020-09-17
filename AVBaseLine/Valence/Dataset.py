import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import Config

class ValenceDataSet(Dataset):
    def __init__(self, transform=None, path=None):
        super(ValenceDataSet, self).__init__()
        self.valence_dataset = pd.DataFrame(pd.read_csv(filepath_or_buffer=path, usecols=Config.valence_need_columns))
        self.length = self.valence_dataset.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.valence_dataset.loc[[idx]].values.tolist()
        img_full_path = os.path.join(Config.manually_annotated_file_prefix, row[0][0])
        # image = Image.fromarray(io.imread(img_full_path))  

        image = Image.open(img_full_path)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'valence': row[0][1]}
        return sample

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = {'image': self.next_data['image'].cuda(non_blocking=True), 'valence': self.next_data['valence'].cuda(non_blocking=True)}
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

def imshow(img):
    img = img / 2 + 0.5
    nping = img.numpy()
    plt.imshow(np.transpose(nping, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    print('Testing Dataset......')
    # record = []
    # for i in range(0, 5):
    #     record.append(i)
    # print(record)
    # record = record[1:]
    # print(record)
    # record.append(6)
    # print(record)
    # print(len(record))
    # input()
    image_transforms = transforms.Compose([
        transforms.Resize((Config.image_resize_height, Config.image_resize_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataSetTrain = ValenceDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_train_path)
    dataSetValid = ValenceDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_validation_path)
    traindataloader = DataLoader(dataSetTrain, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    validdataloader = DataLoader(dataSetValid, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    print(dataSetTrain.length)
    print(dataSetValid.length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for i, data in enumerate(validdataloader, 0):
    #     images, valences = data['image'], data['valence']
    #     print(type(images), type(valences), images.size(), valences.size())
        # images, valences = images.to(device), valences.to(device)
    for i in range(10):
        prefetcher = DataPrefetcher(traindataloader)
        data = prefetcher.next()
        iteration  = 0
        while data is not None:
            iteration  += 1
            images, valences = data['image'].type(dtype=torch.double), data['valence'].type(dtype=torch.double)
            print(iteration, type(images), type(valences), images.size(), valences.size(), images.is_cuda, valences.is_cuda)
            data = prefetcher.next()
            imshow(images[0].cpu())
        # if images.size()[0] < 256:
        #     break
 