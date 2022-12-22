from dataset import *
from utils import *
from model import ResUNet2plus
from resunet_plus import ResUnetPlusPlus
import multiprocessing
import argparse
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingLR,CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast
from loss import BinaryDiceLoss,BCEDice_loss
import wandb
import torch.optim as optim
from tqdm import tqdm
import pprint

sweep_config ={
    'method':'random'
}

metric = {
    'name':'loss',
    'goal':'minimize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer':{
        'values': ['adam','sgd']
    },
    'filters':{
        'values': [16,32,64,128,256]
    },
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config,project='pytorch-sweeps-demo')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataset(batch_size):
    train_dataset = BaseDataset(paths='/mnt/NAS/youngwon/output/sweep_tetst.py',phase='Train')
    train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=multiprocessing.cpu_count()//2,
                    )
    return train_loader

def build_model(filters):
    model = ResUnetPlusPlus(1,filters=filters) 
    return model.to(device)

def build_optimizer(model,optimizer,lr):
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),lr=lr)
    
    return optimizer

def train_epoch(model,loader,optimizer,criterion):
    
    for step, (images,masks,infos) in tqdm(enumerate(loader)):
        images,masks = images.to(device), masks.to(device).long()
        outputs = model(images)
        
        optimizer.zero_grad()
        loss = criterion(outputs,masks)
        optimizer.step()
    print('Success')

pprint.pprint(sweep_config)

def train(config=None):
    with wandb.init(project='Thyroid',name='sweep_test',config=config):
        config = wandb.config
        
        loader = build_dataset(config.batch_size)
        model = build_model(config.filters)
        optimizer = build_optimizer(model,config.optimizer,config.learning_rate)
        criterion = BinaryDiceLoss()

        train_epoch(model,loader,optimizer,criterion)
        
wandb.agent(sweep_id, train, count=5)