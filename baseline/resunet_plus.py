import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride,padding):
        super(ResBlock,self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels,out_channels,3,1,1)
        )
 
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self._initialize_weights()
        
    def forward(self,x):
        identity = x
        x = self.conv_block(x)
        
        return x + self.downsample(identity)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride)
    def forward(self,x):
        return self.upsample(x)

class Attention(nn.Module):
    def __init__(self,input_encoder,input_decoder,out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm3d(input_encoder),
            nn.ReLU(),
            nn.Conv3d(input_encoder,out_channels,3,1,1),
            nn.MaxPool3d((2,2,2))
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm3d(input_decoder),
            nn.ReLU(),
            nn.Conv3d(input_decoder,out_channels,3,1,1)
        )
        self.attn = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels,1,1),
        )
    def forward(self,x1,x2):
        out  = self.encoder(x1) + self.decoder(x2)
        out = self.attn(out)
        return out * x2

class ChannelAttention(nn.Module):
    def __init__(self,channels,reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels,channels//reduction),
            nn.GELU(),
            nn.Linear(channels//reduction,channels)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        b,c,d,h,w = x.size() 
        
        avgpool = self.avg_pool(x).view(b,c)
        maxpool = self.max_pool(x).view(b,c)
        
        avgpool = self.mlp(avgpool)
        maxpool = self.mlp(maxpool)
        
        pool_sum = avgpool + maxpool
        scale = self.sigmoid(pool_sum).view(b,c,1,1,1)
        scale = scale.expand_as(x)
        
        return x * scale
    
class SpatialAttention(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=7):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(2,1,kernel_size=kernel_size,padding=1,bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        out = torch.cat((torch.max(x,dim=1)[0].unsqueeze(1), torch.mean(x,dim=1).unsqueeze(1)),dim=1)
        out = self.spatial(out)
        scale = self.sigmoid(out)
        scale = scale.expand_as(x)
        return x * scale
    
class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=2):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1, 1)
        return x * scale.expand_as(x)

    
class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ASPP,self).__init__()
        self.rate6_block = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,3,1,padding=6,dilation=6,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )
        self.rate12_block = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,3,1,padding=12,dilation=12,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )
        self.rate18_block = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,3,1,padding=18,dilation=18,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )
        self.rate24_block = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,3,1,padding=24,dilation=24,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )
        self.output = nn.Conv3d(out_channels*4,out_channels,1,1,0)
        self._init_weights()
        
    def forward(self,x):
        out1 = self.rate6_block(x)
        out2 = self.rate12_block(x)
        out3 = self.rate18_block(x)
        out4 = self.rate24_block(x)
        return self.output(torch.cat((out1,out2,out3,out4),dim=1))
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
class ResUnetPlusPlus(nn.Module):
    def __init__(self,channel,filters=[16,32,64,128,256]):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv3d(channel,filters[0],kernel_size=3,padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0],filters[0],kernel_size=3,padding=1)
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel,filters[0],kernel_size=3,padding=1),
        )
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.ResBlock1 = ResBlock(filters[0],filters[1],2,1)
        
        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])
        self.ResBlock2 = ResBlock(filters[1],filters[2],2,1)
        
        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.ResBlcok3 = ResBlock(filters[2],filters[3],2,1)
        
        self.bridge = ASPP(filters[3],filters[4])
        
        self.attn1 = Attention(filters[2],filters[4],filters[4])
        self.upsample1 = Upsample(filters[4],filters[4],2,2)
        self.upResBlock1 = ResBlock(filters[4]+filters[2],filters[3],1,1)
        
        self.attn2 = Attention(filters[1],filters[3],filters[3])
        self.upsample2 = Upsample(filters[3],filters[3],2,2)
        self.upResBlock2 = ResBlock(filters[3]+filters[1],filters[2],1,1)
        
        self.attn3 = Attention(filters[0],filters[2],filters[2])
        self.upsample3 = Upsample(filters[2],filters[2],2,2)
        self.upResBlock3 = ResBlock(filters[2]+filters[0],filters[1],1,1)
        
        self.aspp = ASPP(filters[1],filters[0])
        self.fc = nn.Sequential(
            nn.Conv3d(filters[0],1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        pool1 = self.input_layer(x) + self.input_skip(x)
        
        pool2 = self.squeeze_excite1(pool1)
        pool2 = self.ResBlock1(pool2)
        
        pool3 = self.squeeze_excite2(pool2)
        pool3 = self.ResBlock2(pool3)
        
        pool4 = self.squeeze_excite3(pool3)
        pool4 = self.ResBlcok3(pool4)
        
        pool5 = self.bridge(pool4)
        
        attn1 = self.attn1(pool3,pool5)
        unpool1 = self.upsample1(attn1)
        unpool1 = torch.cat((pool3,unpool1),dim=1)
        unpool1 = self.upResBlock1(unpool1)
        
        attn2 = self.attn2(pool2,unpool1)
        unpool2 = self.upsample2(attn2)
        unpool2 = torch.cat((pool2,unpool2),dim=1)
        unpool2 = self.upResBlock2(unpool2)
        
        attn3 = self.attn3(pool1,unpool2)
        unpool3 = self.upsample3(attn3)
        unpool3 = torch.cat((pool1,unpool3),dim=1)
        unpool3 = self.upResBlock3(unpool3)
        
        out = self.aspp(unpool3)
        out = self.fc(out)
        return out
        

if __name__ == '__main__':
    device = torch.cuda.is_available
    m = ResUnetPlusPlus(1)
    input = torch.randn(1,1,64,88,88)
    output = m(input)