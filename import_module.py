import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import scipy as  sp
import pandas as pd
from torchvision import transforms
from torchvision.transforms import *
# import albumentations as A
import random
import time
import os
import glob
from tqdm import tqdm
from pydicom import dcmread
from PIL import Image,ImageDraw,ImageFont
# ImageDraw.ImageDraw.font = ImageFont.truetype("Tests/fonts/FreeMono.ttf")
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
mplstyle.use(['dark_background','fast'])
from sklearn.model_selection import StratifiedKFold,GroupKFold,StratifiedGroupKFold,KFold
import numpy as np
from skimage import io
io.use_plugin("matplotlib")

def getDirs(data_root=None,mask_root=None):
    '''
        data_root='/mnt/NAS/youngwon/handle/data'
        mask_root='/mnt/NAS/youngwon/handle/mask'
    '''
    if data_root is not None: data_dirs = [os.path.join(data_root,i) for i in sorted(os.listdir(data_root)) if 'DS_Store' not in i]
    else: data_dirs = None
        
    if mask_root is not None: mask_dirs = [os.path.join(mask_root,i) for i in sorted(os.listdir(mask_root)) if 'DS_Store' not in i]
    else: mask_dirs= None
    
    return (data_dirs,mask_dirs)

def getDcmList(data_path=None,mask_path=None):
    
    if data_path is not None: data_dcm_paths = [os.path.join(data_path,i) for i in sorted(os.listdir(data_path))]
    else: data_dcm_paths=None
    
    if mask_path is not None: mask_png_paths = [os.path.join(mask_path,i) for i in sorted(os.listdir(mask_path))]
    else: mask_png_paths=None
    
    return (data_dcm_paths,mask_png_paths)

def transform_to_hu(slices,image,axial=0):
    intercept = slices[axial].RescaleIntercept
    slope = slices[axial].RescaleSlope
    hu_image  = slope * image + intercept
    return hu_image

def normalize_image(image,center,width,):
    '''
        window level tuning
    '''
    img_min = center-width//2
    img_max = center+width//2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    window_image = (window_image - img_min) / width
    return window_image

def getDcmImage(dcm_list,center:int=100,width:int=400,scale:int=1,use_interpolation:bool=True,downsample:bool=True):
    slices = [dcmread(dcm) for dcm in dcm_list]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]),reverse=True)
    
    if use_interpolation:
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image = transform_to_hu(slices=slices,image=image,axial=0)
        
        window_image = normalize_image(image,center,width)
        ps = slices[0].PixelSpacing
        ss = slices[0].SliceThickness
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        
        if downsample: interpolation = sp.ndimage.zoom(window_image,[ss/scale,ps[0]/scale,ps[1]/scale])
        else: interpolation = sp.ndimage.zoom(window_image,[scale/ss,scale/ps[0],scale/ps[1]])
        return interpolation
    else: return slices
    

def getNumbers(mask_png_paths):
    numbers = [int(i.split('_')[1][:-4]) for i in mask_png_paths]
    return numbers

def showImage(window,numbers,mask_paths,patient_name,alpha:float=0.3,save=False):
    
    os.makedirs('/mnt/NAS/youngwon/samples/{0}'.format(patient_name),exist_ok=True)
    
    masks, targets = [], []
    index=0
    for i in range(window.shape[0]):
        target = Image.fromarray(window[i])
        if i+1 in numbers:
            mask = Image.open(mask_paths[index])
            masks.append(np.asarray(mask))
            index+=1
        targets.append(np.asarray(target))
    
    index = 0
    for i in range(window.shape[0]):
        plt.figure(figsize=(36,18))
        if i+1 in numbers:
            plt.subplot(1,2,1)
            plt.imshow(targets[i], cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(targets[i],cmap='gray')
            plt.imshow(masks[index], cmap='Reds',alpha=alpha)
            plt.axis('tight')
            index+=1
        else:
            plt.subplot(1,1,1)
            plt.imshow(targets[i],cmap='gray')
            plt.axis('tight')
            
        if save:
            plt.savefig('/mnt/NAS/youngwon/samples/{0}/{1:03d}.png'.format(patient_name,i))
        plt.close()


def dcm_to_png(data_path:str,img_format:str='png'):
    a,b = getDirs(data_path)
    n = len(a)
    for i in range(n):
        c,d = getDcmList(a[i])
        window = getDcmImage(c)
        patient = c[0].split('/')[-2]
        for i in range(window.shape[0]):
            os.makedirs('/mnt/NAS/youngwon/data/{0}'.format(patient),exist_ok=True)
            io.imsave('/mnt/NAS/youngwon/data/{0}/{1:03d}.{2}'.format(patient,i,img_format),window[i])

def bboxVisualize(root='/mnt/NAS/youngwon/test',csv_path='/mnt/NAS/youngwon/handle/nodule_bboxes.csv',mkdir='/mnt/NAS/youngwon/bboxes',img_format:str='png',save=False):
    '''
        root = '/mnt/NAS/youngwon/test
        csv_path = '/mnt/NAS/youngwon/handle/nodule_bboxes.csv
    '''
    df = pd.read_csv(csv_path)
    bboxes = df[['x_min','y_min','x_max','y_max']].apply(list).values
    title = df[['subject','slice_location']].astype(str).values

    db = {}
    for a,b in zip(title,bboxes):
        key = '/'.join(a)
        if key not in db: db[key] = [b]
        else: db[key].append(b)
        
    a,b = getDirs(data_root=root)
    for key,value in db.items():
        temp = key.split('/')
        os.makedirs('{0}/{1}'.format(mkdir,temp[0]),exist_ok=True)
        img_path = '{0}/{1}/{2:03d}.{3}'.format(root,temp[0],int(temp[1])-1,img_format)
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        for i in range(len(value)):
            draw.rectangle((db[key][i]), outline=(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)), width = 1)
        f,ax = plt.figure(figsize=(18,18)) , plt.subplot(1,1,1)
        ax.imshow(img)
        ax.axis('tight')
        
        if save: plt.savefig('{bboxes}/{0}/{1}.{2}'.format(mkdir,temp[0],temp[1],img_format))
        plt.close()

def make3Dlabel(data_dir,output_dir):
    paths = [os.path.join(data_dir,p) for p in os.listdir(data_dir)]
    imgs = [np.asarray(Image.open(path)) for path in paths]
    labels = np.stack(imgs)
    return labels

def make3Ddata(data_root=None,mask_root=None,save_root=None,label=False):
    
    data_paths, mask_paths = getDirs(data_root,mask_root) #전체 paths를 가져옵니다.

    os.makedirs('{0}'.format(save_root),exist_ok=True)
    if not label:
        for i in tqdm(range(len(data_paths))):
            patient_name = data_paths[i].split('/')[-1]
            paths,_ = getDcmList(data_paths[i]) #환자dcm  path
            window = getDcmImage(paths)
            np.save('{0}/{1}'.format(save_root,patient_name),window)
    else:
        infos = []
        for i in tqdm(range(len(data_paths))):
            paths,_ = getDcmList(data_paths[i]) #환자의 dcm 파일 경로를 가져옵니다.
            dcm = dcmread(paths[0]) #dcm 파일의 첫 번째를 가져와 읽습니다.
            size = dcm.pixel_array.shape #dcm 파일을 numpy_array로 바꾸어줍니다.
            depths = len(paths) # axial의 개수를 계산합니다.
            ps = dcm.PixelSpacing # pixel spacing 값을 가져옵니다.
            ss = dcm.SliceThickness # slice thickness 값을 가져옵니다.
            infos.append(((depths,*size),ps,ss)) # 불러온 헤더 값들을 infos에 저장합니다.
            
        for i in tqdm(range(len(mask_paths))):
            patient_name = mask_paths[i].split('/')[-1]  + '_mask' # 환자 id를 지정
            mask,ps,ss = np.zeros(tuple(infos[i][0])),infos[i][1],infos[i][2] # 3d mask정보를 임시로 만들어 둡니다.
            paths = [os.path.join(mask_paths[i],p) for p in os.listdir(mask_paths[i])] # mask 경로를 모두 가져옵니다.
            paths.sort()
            
            location = getNumbers(paths) #3d mask 배열에서 label로 주어진 mask의 정보를 계산합니다.
            location.sort()
            
            imgs = [np.asarray(Image.open(path)) for path in paths] # mask 이미지들을 모두 읽어들이고 배열에 저장합니다.
            
            for loc,img in zip(location,imgs): #3d mask의 axial 번호를 location으로 특정하고 img를 삽입합니다.
                mask[loc-1] = img
            mask = mask[::-1]
            scale=1

            mask = sp.ndimage.zoom(mask,[ss/scale,ps[0]/scale,ps[1]/scale], order=1) 
            #pixel spacing, slice thickness를 1mm로 모두 정규화해줍니다.
            #대경님이 알려주신 방법으로 order=1로 넣었을 때 마스크가 비교적 깨끗하게 나옵니다.
            
            mask[mask>0] = 1. # 0 보다 크면 모두 1로 초기화
            mask[mask<0] = 0. # 0 보다 작으면 모두 0으로 초기화  -> 이진화
            np.save('{0}/{1}'.format(save_root,patient_name),mask) #원하는 경로에 저장
            
def center_crop(data_root,mask_root,save_root,width,height,depths:tuple):
    
    data_paths,mask_paths = getDirs(data_root,mask_root)
    os.makedirs('{0}/{1}'.format(save_root,'crop_all'),exist_ok=True)
    os.makedirs('{0}/{1}'.format(save_root,'crop_label'),exist_ok=True)
    
    for idx in tqdm(range(len(data_paths))):
        
        patient_name = data_paths[idx].split('/')[-1]
        data3d = np.load(data_paths[idx])
        mask3d = np.load(mask_paths[idx])
        d_size = data3d.shape
        m_size = mask3d.shape
        if d_size[0] < max(depths) or d_size[1] < width or d_size[2] < height:
            print('Size error')
            print(d_size)
            return
        if m_size[0] < max(depths) or m_size[1] < width or m_size[2] < height:
            print('Size error')
            print(m_size)
            return

        croped_imgs = []
        croped_masks = []
        for depth in tqdm(range(depths[0],depths[1])):
            img = data3d[depth] #axial
            mask = mask3d[depth]
            img_center = (img.shape[0]//2,img.shape[1]//2) # w,h
            mask_center = (mask.shape[0]//2,mask.shape[1]//2)# w,h
            crop_img = img[ img_center[0] - width//2 : img_center[0] + width//2 , img_center[1] - height//2 : img_center[1] + height//2]
            crop_mask = mask[ mask_center[0] - width//2 : mask_center[0] + width//2, mask_center[1] - height//2 : mask_center[1] + height//2]
            croped_imgs.append(crop_img)
            croped_masks.append(crop_mask)
        
        croped_imgs = [np.stack(i) for i in croped_imgs]
        croped_masks = [np.stack(i) for i in croped_masks]
            
        np.save('{0}/{1}/{2}'.format(save_root,'crop_all',patient_name),croped_imgs)
        np.save('{0}/{1}/{2}'.format(save_root,'crop_label',patient_name),croped_masks)
        print(f'Success the npy file! to {save_root}/crop_all/{patient_name} ...', idx)
        print(f'Success the npy file! to {save_root}/crop_label/{patient_name} ...', idx)
        croped_imgs = []
        croped_masks = []
        
        
def make_csv_all(data_root,mask_root):
    columns = ['patient_id','data','mask']
    df = pd.DataFrame(columns=columns)

    patient_id = [path[:-4] for path in os.listdir(data_root)]
    data_paths = [os.path.join(data_root,path) for path in os.listdir(data_root)]
    mask_paths = [os.path.join(mask_root,path) for path in os.listdir(mask_root)]
    
    df = pd.DataFrame(list(zip(patient_id,data_paths,mask_paths)),columns=columns)
    df.to_csv('/mnt/NAS/youngwon/data/all.csv',index=False)
    
def test():

    a,b = getDirs()
    n = len(a)
    for i in range(n):
        c,d = getDcmList(a[i],b[i])
        window = getDcmImage(c)
        numbers = getNumbers(d)
        patient = d[0].split('/')[-2]
        showImage(window,numbers,d,patient)
        
