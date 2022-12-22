import torch
import torch.nn as nn
from focalnet import FocalNet
from torchsummary import summary

def conv_block(in_ch,out_ch,k_size,stride,padding,dilation=1,relu=True):
    block = []
    block.append(nn.Conv2d(in_ch,out_ch,k_size,stride,padding,dilation,bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)

class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self):
        super().__init__()
        #1x1 conv
        self.block1 = conv_
class DeepLabV3_FocalNet():
    def __init__(self) -> None:
        self.backbone = FocalNet()
        self.aspp = 

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    # input = torch.randn(1,1,256,256,256)
    summary(model, (1,64,128,128))
    # out = model(input)
    # print(out.shape)