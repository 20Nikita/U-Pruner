import torch
from torch import nn
import torch.nn.functional as F

from resnet34 import resnet34

def replace_strides_with_dilation(module, dilation_rate):
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)
            
class resnet18_encoder(resnet34):
    def __init__(self):
        super().__init__()
        del self.avgpool
        del self.fc
        for mod in self.layer3.modules():
            replace_strides_with_dilation(mod, 2)
        for mod in self.layer4.modules():
            replace_strides_with_dilation(mod, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ),
            nn.Sequential( 
                nn.Conv2d(512, 256, kernel_size=3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ),
            nn.Sequential( 
                nn.Conv2d(512, 256, kernel_size=3, padding=24, dilation=24, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ),
            nn.Sequential( 
                nn.Conv2d(512, 256, kernel_size=3, padding=36, dilation=36, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                )
        ])
        self.project = nn.Sequential(
            nn.Conv2d(5 * 256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        size = x.shape[-2:]
        x = self.convs[-1](x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        res.append(x)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.encoder = resnet18_encoder()
        self.decoder = nn.Sequential(
            ASPP(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8))

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks