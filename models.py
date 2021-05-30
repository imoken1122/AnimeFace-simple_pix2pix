import torch as th
from torch import nn
import numpy as np
class Generator_Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1,64,4,stride=2,padding=1)
        self.enc2 = self.conv_block(64,128, )
        self.enc3 = self.conv_block(128,256)
        self.enc4 = self.conv_block(256,512,)
        self.enc5 = self.conv_block(512,512,)

        self.inner_vec = nn.Sequential(nn.LeakyReLU(0.1,inplace=True), nn.Conv2d(512,512,4,2,1))
        
        self.dec0 = self.conv_block(512,512,upsample=True,dropout=False)
        self.dec1 = self.conv_block(512,512,upsample=True,)
        self.dec2 = self.conv_block(512 + 512,512,upsample=True,)
        self.dec3= self.conv_block(512+512,256,upsample=True,)
        self.dec4= self.conv_block(256+256,128,upsample=True,)
        self.dec5= self.conv_block(128+128,64,upsample=True,)
        self.output = nn.Sequential(nn.ReLU(inplace=True), nn.ConvTranspose2d(64+64,3,4,2,1),nn.Tanh())

    def conv_block(self,in_ch, out_ch, kernel_size=4,stride=2,upsample = False,dropout = True):
        layers = []
        if not upsample:
            layers.append(nn.LeakyReLU(0.1,inplace=True))
            layers.append(nn.Conv2d(in_ch, out_ch,kernel_size,stride=stride,padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
        else:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(in_ch,out_ch,kernel_size,stride=stride,padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            if dropout : layers.append(nn.Dropout(0.25))
        return nn.Sequential(*layers)
    def forward(self,x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc5(x5)
        inner_vec = self.inner_vec(x6)


        out = self.dec0(inner_vec)
        out = self.dec1(out)
        out = self.dec2(th.cat([out,x5],1))
        out = self.dec3(th.cat([out,x4],1))
        out = self.dec4(th.cat([out,x3],1))
        out = self.dec5(th.cat([out,x2],1))
        out = self.output(th.cat([out,x1],1))
        return out

class Generator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc1 = self.conv_block(1,32,5)                  # [bs, 32, 256, 256]
        self.enc2 =  self.conv_block(32,64,3,pool_kernel = 4) #[bs, 64, 64, 64]
        self.enc3 = self.conv_block(64,128,3,pool_kernel =2) # [bs, 128, 32, 32]
        self.enc4 = self.conv_block(128,256,3,pool_kernel =2) #[bs, 256, 16, 16]
        self.enc5 = self.conv_block(256,512,3,pool_kernel =2) #[ 8, 512, 8, 8]
    
        self.dec1 = self.conv_block(512,256,3,pool_kernel =-2) # [bs, 256, 16, 16]
        self.dec2 = self.conv_block(256+256,128,3,pool_kernel =-2) #[bs, 128, 32, 32]
        self.dec3 = self.conv_block(128+128,64,3,pool_kernel =-2) #([bs, 64, 64, 64]
        self.dec4 = self.conv_block(64+64,32,3,pool_kernel =-4) #[bs, 32, 256, 256])
        self.output = nn.Sequential(nn.Conv2d(32+32,3,5,padding=2 ), nn.Tanh()) # [bs, 3, 256, 256]

    def conv_block(self,in_ch,out_ch,kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if  pool_kernel>0:
                layers.append(nn.AvgPool2d(pool_kernel)) 
            else:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))

        layers.append(nn.Conv2d(in_ch,out_ch,kernel_size, padding = (kernel_size-1)//2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self,x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        
        y1 = self.dec1(x5)
        y2 = self.dec2(th.cat([y1,x4],dim = 1))
        y3 = self.dec3(th.cat([y2,x3],dim = 1))
        y4 = self.dec4(th.cat([y3,x2],dim = 1))
        output = self.output(th.cat([y4,x1],dim = 1))
        return output


class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super(Discriminator, self).__init__()
        self.cv1 = self.conv_block(in_ch,32,3,num = 1) # concat sketch+gen img at ch_dim
        self.cv2 = self.conv_block(32,64,3,num = 1) # concat sketch+gen img at ch_dim
        self.cv3 = self.conv_block(64,128,pool_kernel=2)
        self.cv4 = self.conv_block(128,256,pool_kernel=2)
        self.output = nn.Conv2d(256,1,kernel_size=1) # [bs,1,32,32]


    def conv_block(self,in_ch, out_ch, kernel_size=3, pool_kernel=None,num = 2):
        layers = []
        for i in range(num):
            if i==0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i ==0 else out_ch ,out_ch, kernel_size,padding= (kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.cv4(self.cv3(self.cv2(self.cv1(x))))
        return self.output(out)


"""
from torchsummary import summary
model = Generator()
summary(model,(1,256,256))
"""