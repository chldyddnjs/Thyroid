import os
import pydicom
path = '/mnt/NAS/youngwon/handle/data'
paths = []
for i in os.listdir(path):
    paths.append(os.path.join(path,i))

from pydicom.data import get_testdata_file
from pydicom import dcmread,read_file
import numpy as np

fpath = get_testdata_file("CT_small.dcm")

ds = dcmread(fpath)

with open(fpath,'rb') as infile:
    ds = dcmread(infile)

ds = read_file(fpath)
# print(ds)

img = ds.pixel_array

# print(img)
#image location 기준으로 정렬

paths.sort()
print(paths)
info = os.listdir(paths[0])
slices = [dcmread(os.path.join(paths[0],i)) for i in info]
slices.sort(key=lambda x: x.SliceLocation)
image = np.stack([s.pixel_array for s in slices])
#image.decompress('gdcm')
#pip install gdcm
import matplotlib
from skimage import io
matplotlib.use('Agg')

import matplotlib.pyplot as plt
io.use_plugin("matplotlib")
for i in range(image.shape[0]):
    io.imsave( f"target{i}.png", image[i]) #axial view z축 절단면을 눕힌거
# plt.imshow(image[80])
# plt.show()
# plt.imwrite('test', '.png')

