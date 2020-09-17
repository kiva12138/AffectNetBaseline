import torch
from torch.utils.tensorboard import SummaryWriter


files = [r'D:\Coding\NetworkWithPytorch\AffectNet\CategoricalBaseLine\Draw Valid Loss\Imbalance_log.out',
    r'D:\Coding\NetworkWithPytorch\AffectNet\CategoricalBaseLine\Draw Valid Loss\Weighted_log.out']
log_path = [r'D:\Coding\NetworkWithPytorch\AffectNet\CategoricalBaseLine\Logdir\Valid Loss\Imbalance',
    r'D:\Coding\NetworkWithPytorch\AffectNet\CategoricalBaseLine\Logdir\Valid Loss\Weighted']
tag_str = 'Valid Epoch'

for i in range(2):
    writer = SummaryWriter(log_path[i])
    fin = open(files[i], mode='r')
    lines = fin.readlines()
    epoch = 1
    for line in lines:
        if  tag_str in line:
            target = line.split(',')[0].split(' ')[-1]
            # print(float(target))
            writer.add_scalar('Valid Loss', float(target), epoch)
            epoch += 1
    writer.close()
    print(log_path[i], ' Over')
