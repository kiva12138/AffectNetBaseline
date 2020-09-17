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

class ExpressionDataSet(Dataset):
    def __init__(self, transform=None, path=None):
        super(ExpressionDataSet, self).__init__()
        self.dataset = pd.DataFrame(pd.read_csv(filepath_or_buffer=path, usecols=Config.classify_need_columns))
        self.length = self.dataset.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.dataset.loc[[idx]].values.tolist()
        img_full_path = os.path.join(Config.manually_annotated_file_prefix, row[0][0])
        # image = Image.fromarray(io.imread(img_full_path))  

        image = Image.open(img_full_path)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'expression': row[0][1]}
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
            self.next_data = {'image': self.next_data['image'].cuda(non_blocking=True), 'expression': self.next_data['expression'].cuda(non_blocking=True)}
            
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
        transforms.RandomCrop(size=(Config.image_crop_height, Config.image_crop_height)),
        # transforms.FiveCrop((Config.image_crop_height, Config.image_crop_height)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(norm) for norm in norms]))
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # dataSetValid = ExpressionDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_validation_path)
    dataSetValid = ExpressionDataSet(transform=image_transforms, path=Config.manually_annotated_file_list_train_path)
    validdataloader = DataLoader(dataSetValid, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(dataSetValid.length)

    prefetcher = DataPrefetcher(validdataloader)
    data = prefetcher.next()
    iteration  = 0
    while data is not None:
        iteration  += 1
        images, expressions = data['image'].type(dtype=torch.double), data['expression'].type(dtype=torch.double)
        print(iteration, images.size(), expressions.size(), images.is_cuda, expressions.is_cuda)
        data = prefetcher.next()
        # input()
        # for img in images:
        #     imshow(img.cpu())
        
 