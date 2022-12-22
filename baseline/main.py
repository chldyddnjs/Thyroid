from dataset import *
from utils import *
from resunet_plus import ResUnetPlusPlus
import multiprocessing
import argparse
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingLR,CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast
from loss import BinaryDiceLoss,BCEDice_loss
import wandb
import metrics
import random
from tqdm import tqdm
import torch.optim as optim
import os

def save_model(model, saved_dir, file_name='resUnetPlusPlus_best_model.pth'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model.module.state_dict(), output_path)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# -- settings
category_names = ['Thyroid']
category_dict = {0:'Thyroid'}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def build_model(filters):
    model = ResUnetPlusPlus(1,filters=filters) 
    return model

def build_optimizer(model,optimizer,lr):
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999))
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.999))
    
    return optimizer

def train(train_paths,valid_paths,args,sweep_config=None):
    seed_everything(args.seed)
    best_mdice = 0
    train_dataset = BaseDataset(train_paths,phase='Train')
    valid_dataset = BaseDataset(valid_paths,phase='Val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count()//2,
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count()//2,
        )
    
    model = build_model(filters=[16,32,64,128,256])
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    criterion = BinaryDiceLoss()
    optimizer = build_optimizer(model,'adam',args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=args.T_0, eta_min=0 ,last_epoch=-1)
    # scheduler = StepLR(optimizer,0.01, gamma=0.5)
    
    #--wandb setting
    config = args.__dict__.copy()
    
    wandb_name = f'{args.user}_{args.exp_name}'
    
    wandb.init(project='thyroid',name=wandb_name,config=config)
    wandb.watch(model)
    
    for epoch in range(args.epochs):
        model.train()
        
        train_mask_list = []
        train_pred_list = []
        valid_mask_list = []
        valid_pred_list = []
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        
        for step, (images,masks,infos) in pbar:
            
            # run training and validation
            # logging accuracy and loss
            train_loss = metrics.MetricTracker()
            train_dice = metrics.MetricTracker()
            # iterate over data
            
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs,masks)
            
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.mean().backward()
            optimizer.step()
            
            train_dice_score = metrics.dice_coeff(outputs, masks)
            train_loss.update(loss.data.item(), outputs.size(0))
            train_dice.update(train_dice_score, outputs.size(0))
            
            lr_rate = optimizer.param_groups[0]['lr']
            pbar.set_description(f'Epoch [{epoch+1}/{args.epochs}], Step[{step+1}/{len(train_loader)}],Loss:{round(loss.item(),4)},train_dice:{round(train_dice_score,4)}, lr:{round(lr_rate,4)}')
            
            wandb.log({
                'learning_rate': lr_rate,
            })
            
            outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            for i in range(args.batch_size//16):
                origin = images[i].squeeze()
                pred = outputs[i].squeeze()
                target = masks[i].squeeze()
                patient_name = infos[i]
                for j in range(len(origin)//16):
                    wandb_media_pred = wandb.Image(origin[j],caption=f'study:{patient_name}', masks={
                            "predictions" : {
                                "mask_data" : pred[j],
                                "class_labels" : category_dict
                                },
                            # "ground_truth" : {
                            #     "mask_data" : target[0],
                            #     "class_labels" : category_dict}
                            })
                    wandb_media_mask = wandb.Image(origin[j],caption=f'study:{patient_name}', masks={
                            # "predictions" : {
                            #     "mask_data" : pred[0],
                            #     "class_labels" : category_dict
                            #     },
                            "ground_truth" : {
                                "mask_data" : target[j],
                                "class_labels" : category_dict
                                }
                            })
                    train_pred_list.append(wandb_media_pred)
                    train_mask_list.append(wandb_media_mask)


        if epoch%20 == 0:
            np.save(f'/mnt/NAS/youngwon/output/checkImg/{epoch}_pred',outputs)
            np.save(f'/mnt/NAS/youngwon/output/checkImg/{epoch}_mask',masks)

        wandb.log({
                    "train/loss" : train_loss.avg,
                    "train/mdice": train_dice.avg
                    })
        wandb.log({"train_pred_media" : train_pred_list})
        wandb.log({"train_mask_media" : train_mask_list})

        valid_acc = metrics.MetricTracker()
        valid_loss = metrics.MetricTracker()
        valid_dice = metrics.MetricTracker()
        model.eval()
        with torch.no_grad():
            for idx, (images,masks,infos) in enumerate(tqdm(valid_loader, desc="validation")):

                # get the inputs and wrap in Variable
                images = images.to(device)
                masks = masks.to(device).long()

                # forward
                # prob_map = model(inputs) # last activation was a sigmoid
                # outputs = (prob_map > 0.3).float()
                outputs = model(images)
                # outputs = torch.nn.functional.sigmoid(outputs)
                loss = criterion(outputs, masks)

                val_dice_score = metrics.dice_coeff(outputs, masks)
                valid_loss.update(loss.data.item(), outputs.size(0))
                valid_dice.update(val_dice_score, outputs.size(0))
                
                outputs = outputs.detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()

                for i in range(len(images)//8):
                    origin = images[i].squeeze()
                    pred = outputs[i].squeeze()
                    target = masks[i].squeeze()
                    patient_name = infos[i]
                    for j in range(len(origin)//16):
                        valid_pred_media = wandb.Image(origin[j],caption=f'study:{patient_name}', masks={
                            "predictions" : {
                                "mask_data" : pred[j],
                                "class_labels" : category_dict
                                }
                            # "ground_truth" : {
                            #     "mask_data" : target[0],
                            #     "class_labels" : category_dict}
                            })
                        valid_mask_media = wandb.Image(origin[j],caption=f'study:{patient_name}', masks={
                            # "predictions" : {
                            #     "mask_data" : pred[0],
                            #     "class_labels" : category_dict
                            #     }
                            "ground_truth" : {
                                "mask_data" : target[j],
                                "class_labels" : category_dict
                                }
                            })
                        valid_mask_list.append(valid_mask_media)
                        valid_pred_list.append(valid_pred_media)
                    
            wandb.log({
                        "valid/mdice": valid_dice.avg,
                        "valid/loss": valid_loss.avg,
                    })
            wandb.log({"valid_pred_media" : valid_pred_list})
            wandb.log({"valid_mask_media" : valid_mask_list})
            if valid_dice.avg > best_mdice:
                best_mdice = valid_dice.avg
                save_model(model, args.saved_dir, f'epoch{epoch+1:04d}_mdice{str(round(best_mdice,4)).replace(".","")}.pth')
            print("Validation Loss: {:.4f} Dice: {:.4f}".format(valid_loss.avg,valid_dice.avg))
        scheduler.step()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42,help='random seed (default:42)')
    parser.add_argument('--epochs',type=int, default=300,help='number of epochs to train')
    parser.add_argument('--image_size',type=tuple, default=(128,128),help='number of epochs to train')
    parser.add_argument('--batch_size',type=int,default=16,help='number of batchs to train')
    parser.add_argument('--threshold',type=float,default=0.5,help='number of batchs to train')
    parser.add_argument('--momentum',type=int,default=0.9,help='number of momentum to train')
    parser.add_argument('--scheduler',type=str,default='CosineAnnealingLR',help='scheduler')
    parser.add_argument('--T_max',type=int,default=20,help='CosinannealingLR frequence cycle')
    parser.add_argument('--T_0',type=int,default=20,help='CosinannelingWarmRestarts')
    parser.add_argument('--lr',type=float,default=0.005,help='learning_rate')
    parser.add_argument('--criterion', type=str, default='BinaryDiceLoss', help='criterion type (default: cross_entropy)')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
    parser.add_argument('--exp_name', type=str, default='9')
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--user', type=str,default='youngwon')
    parser.add_argument('--display_fname', type=int, default=1)
    parser.add_argument('--saved_dir',type=str,default='/mnt/NAS/youngwon/output/best_pth')
    args = parser.parse_args()
    
    # main()
    train_paths = f'/mnt/NAS/youngwon/data/crop_train{args.fold}.csv'
    valid_paths = f'/mnt/NAS/youngwon/data/crop_val{args.fold}.csv'
    train(train_paths,valid_paths,args)