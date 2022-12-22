import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from dataset import BaseDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import multiprocessing
from resunet_plus import ResUnetPlusPlus
import numpy as np
from tqdm import tqdm
import metrics
import wandb
from copy import deepcopy


category_dict = {0:'thyroid'}

def load_model(model_path,device):
    model = ResUnetPlusPlus(1,filters=[16,32,64,128,256])
    model.to(device)
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint)
    return model

def test(model,data_loader,device,args):
    
    size = 128
    
    model.eval()

    test_pred_list = []
    test_mask_list = []
    # -- wandb setting
    exp_name = args.model + '_' + args.model_path.split('/')[-1].split('.')[0]
    wandb_name = f'{args.user}_{exp_name}'
    wandb.init(project = 'seg_inference', name = wandb_name)
    wandb.watch(model, log=None)
    
    with torch.no_grad():
        for step, (images,masks,infos) in enumerate(tqdm(data_loader)):
            #inference (128 x 128)
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            dice_score = metrics.dice_coeff(deepcopy(outputs), masks)
            
            images = images.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            wandb.log({
                'Dice':dice_score
            })
            for i in range(len(images)):
                origin = images[i].squeeze()
                pred = outputs[i].squeeze()
                target = masks[i].squeeze()
                patient_name = infos[i]
                os.makedirs('/mnt/NAS/youngwon/inference/fold{0}/{1}'.format(args.fold,patient_name),exist_ok=True)
                plt.figure(figsize=(18,18))
                for j in range(len(origin)):
                    plt.subplot(1,3,1)
                    plt.imshow(origin[j],cmap='gray')
                    plt.subplot(1,3,2)
                    plt.imshow(pred[j],cmap='gray')
                    plt.subplot(1,3,3)
                    plt.imshow(target[j],cmap='gray')
                    plt.savefig('/mnt/NAS/youngwon/inference/fold{0}/{1}/output{2:03d}.png'.format(args.fold,patient_name,j))
                    plt.close()
            
            
                    # test_pred_list.append(wandb_media_pred)
                    # test_mask_list.append(wandb_media_mask)
            
    return test_pred_list
def inference(test_path,args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = load_model(args.model_path,device).to(device)
    
    # dataset
    test_dataset = BaseDataset(paths=test_path,phase='Test')
    test_loader = DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=multiprocessing.cpu_count()//2,
                        )
    test(model,test_loader,device,args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128, 128))
    parser.add_argument('--model', type=str, default='ResUNet++', help='model type (default: BaseModel)')
    parser.add_argument('--fold',type=int,default=0)
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))

    # segmentation
    parser.add_argument('--dataset_path', type=str, default='s/opt/ml/input/data')
    parser.add_argument('--model_path', type=str, default='/mnt/NAS/youngwon/output/best_pth/epoch0138_mdice08755.pth')

    parser.add_argument('--user', type=str,default='youngwon')

    args = parser.parse_args()
    test_path = '/mnt/NAS/youngwon/data/crop_test.csv'
    inference(test_path,args)
    
    