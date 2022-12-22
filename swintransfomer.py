import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x,window_size):
    """_summary_

    Args:
        x:(B,H,W,C)
        window_size (int): window size
    returns:
        windows: (num_windows*B, window_size,window_size,C)
    """
    B,H,W,C = x.shape
    x = x.view(B , H // window_size,window_size, W // window_size, window_size , C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view( -1, window_size, window_size, C)
    return windows

def window_reverse(windows,window_size,H,W):
    """_summary_

    Args:
        windows: 
        window_size (_type_): _description_
        H (_type_): _description_
        W (_type_): _description_
    """
    B = int(windows.shape[0] / (H*W/ window_size / window_size))
    x = windows.view(B,H // window_size, W//window_size, window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    
    return x

# class WindowAttention(nn.Module):
    
    