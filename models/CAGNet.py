import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans import ChannelTransformer
# from models.block.Drop import DropBlock
from .Drop import DropBlock

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
    
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
    
class ConvBatchNorm_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, padding=0)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels,  nb_Conv,activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.conv_1x1=ConvBatchNorm_1x1(in_channels,in_channels//2)
         

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x=self.conv_1x1(skip_x)
       
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)
    
class _upsample_like(nn.Module):
     def __init__(self, in_channels, out_channels,activation='ReLU'):
        super().__init__()
        self.conv_1x1=ConvBatchNorm_1x1(in_channels,out_channels)

     def forward(self, src,tar):
         src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
         src=self.conv_1x1(src)
         return src

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),stride=(stride, stride), bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.extract(x)
        return x

class Conv1Relu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv1Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.extract(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.block(x) + residual
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        out_channels = int(in_channels / 2)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))
        self.dim_reduction = Conv3Relu(out_channels * 5, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)
        out = self.dim_reduction(torch.cat((feat0, feat1, feat2, feat3, feat4), 1))

        return out

class GAmodule(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)
        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
        self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)
        self.final_Conv = Conv3Relu(inplanes * 4, inplanes)
        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.expand_field = ASPP(inplanes * 8)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)
        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])
        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 8
        #ASPP
        change4 = self.expand_field(change4)
        change3_2 = self.stage4_Conv_after_up(self.up(change4))
        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1))
        change2_2 = self.stage3_Conv_after_up(self.up(change3))
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1))
        change1_2 = self.stage2_Conv_after_up(self.up(change2))
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1))
        change4 = self.stage4_Conv3(F.interpolate(change4, size=(change1_h, change1_w),
                                                  mode='bilinear', align_corners=True))
        change3 = self.stage3_Conv3(F.interpolate(change3, size=(change1_h, change1_w),
                                                  mode='bilinear', align_corners=True))
        change2 = self.stage2_Conv3(F.interpolate(change2, size=(change1_h, change1_w),
                                                  mode='bilinear', align_corners=True))
        [change1, change2, change3, change4] = self.drop([change1, change2, change3, change4])
        change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))

        return change

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.head = nn.Sequential(Conv3Relu(in_channels, inter_channels), nn.Dropout(0.2), nn.Conv2d(inter_channels, out_channels, (1, 1)))

    def forward(self, x):
        return self.head(x)

class CAGNet(nn.Module):
    def __init__(self, config,n_channels=3,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.CA = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.GA = GAmodule(in_channels)
        self.head = FCNHead(in_channels, 2)

    def forward(self, xA, xB):
        _, _, h_input, w_input = xA.shape
        out_size = (h_input, w_input)
        xA = xA.float()
        x1A = self.inc(xA)
        x2A = self.down1(x1A)
        x3A = self.down2(x2A)
        x4A = self.down3(x3A)
        xB = xB.float()
        x1B = self.inc(xB)
        x2B = self.down1(x1B)
        x3B = self.down2(x2B)
        x4B = self.down3(x3B)
        #CA module
        x1A, x2A, x3A, x4A, att_weightsA = self.CA(x1A, x2A, x3A, x4A)
        x1B, x2B, x3B, x4B, att_weightsB = self.CA(x1B, x2B, x3B, x4B)
        ms_feats = x1A, x2A, x3A, x4A, x1B, x2B, x3B, x4B
        #GA module
        change = self.GA(ms_feats)
        out = F.interpolate(self.head(change), size=out_size, mode='bilinear', align_corners=True)

        return out