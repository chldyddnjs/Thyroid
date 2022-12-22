# Thyroid
3D Binary Segmentation for Thyroid

# Introduction
ResUnet++을 이용한 Thyroid segmenation

국립 암센터의 Thyroid dicom data를 3D Volumn으로 전처리하고 3D로 remodeling한 ResUnet으로 학습

총 250개의 데이터 중 160개,40개,50개를 각각 training set,validation set, test set으로 활용

# Metrix - Dice score
label(1) 영역과 background(0) 영역의 비율이 상당히 불균형한것을 확인하였고 평가지표로서 Dice score가 적합

# Wandb for training 
![image](https://user-images.githubusercontent.com/48711155/209058039-d0623097-531b-423b-ad19-a88943532dba.png)
![image](https://user-images.githubusercontent.com/48711155/209058180-e525305c-1d21-4285-8dbf-1f08cd27b957.png)

# Inference Result
![image](https://user-images.githubusercontent.com/48711155/209059155-f6be5163-897e-49dc-8d20-001b596449b6.png)

# samples
![output003](https://user-images.githubusercontent.com/48711155/209066799-512e0ead-e35b-4ee9-acce-0c80f416542e.png)
![output019](https://user-images.githubusercontent.com/48711155/209066865-dfc61269-c612-4c9f-9208-b61eb602e807.png)
![output024](https://user-images.githubusercontent.com/48711155/209066889-f1c27bea-0035-44cd-96e5-872fed179a4e.png)
![output029](https://user-images.githubusercontent.com/48711155/209066898-dd26b525-76da-406d-8793-b11c3605985c.png)
![output032](https://user-images.githubusercontent.com/48711155/209066816-38b9c178-8acc-4a73-9d47-6dcc70aa8857.png)
![output033](https://user-images.githubusercontent.com/48711155/209057435-066ffe7e-06f6-4513-8415-71d9b8ea68f6.png)
![output044](https://user-images.githubusercontent.com/48711155/209066912-b91b0c33-b1e9-4af4-a881-fdf744ae4c81.png)
![output054](https://user-images.githubusercontent.com/48711155/209066917-a1a5ba6b-8aac-4b29-988f-bf6ed98a4236.png)
![output059](https://user-images.githubusercontent.com/48711155/209066925-6d85fe8c-98b3-4a44-ad26-9550fc58b1e3.png)
