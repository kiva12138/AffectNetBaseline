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

class ArousalDataSet(Dataset):
    def __init__(self, transform=None, path=None):
        super(ArousalDataSet, self).__init__()
        self.arousal_dataset = pd.DataFrame(pd.read_csv(filepath_or_buffer=path, usecols=Config.arousal_need_columns))
        self.length = self.arousal_dataset.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.arousal_dataset.loc[[idx]].values.tolist()
        img_full_path = os.path.join(Config.manually_annotated_file_prefix, row[0][0])
        # image = Image.fromarray(io.imread(img_full_path))  

        image = Image.open(img_full_path)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'arousal': row[0][1]}
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
            self.next_data = {'image': self.next_data['image'].cuda(non_blocking=True), 'arousal': self.next_data['arousal'].cuda(non_blocking=True)}
            
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
    image_transforms = transforms.Compose([
        transforms.Resize((Config.image_resize_height, Config.image_resize_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataSetTrain = ArousalDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_train_path)
    dataSetValid = ArousalDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_validation_path)
    traindataloader = DataLoader(dataSetTrain, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    validdataloader = DataLoader(dataSetValid, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    print(dataSetTrain.length, dataSetValid.length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(10):
        prefetcher = DataPrefetcher(validdataloader)
        data = prefetcher.next()
        iteration  = 0
        while data is not None:
            iteration  += 1
            images, arousals = data['image'].type(dtype=torch.double), data['arousal'].type(dtype=torch.double)
            print(iteration, type(images), type(arousals), images.size(), arousals.size(), images.is_cuda, arousals.is_cuda,\
                images[0][0][0], arousals[0])
            data = prefetcher.next()
            input()
 