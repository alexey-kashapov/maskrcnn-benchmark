import torch
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.checkpoint import checkpoint

import torchvision.transforms
from PIL import Image

class MyNet(nn.Module):
    def __init__(self, cfg):

        super(MyNet, self).__init__()
        block = Bottleneck
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)

        self.conv_fuse = nn.Conv2d(2048, 1024, kernel_size=1,stride=1,padding=0,bias=False)
        self.bn_fuse = nn.BatchNorm2d(1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_downsample(self, rgb, depth):

        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)

        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        x = self.maxpool(x)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        depth = self.layer1_d(depth)

        # block 2
        x = self.layer2(x)
        depth = self.layer2_d(depth)

        # block 3
        x = self.layer3(x)
        depth = self.layer3_d(depth)

        fuse = torch.cat((x,depth),1)
        fuse = self.conv_fuse(fuse)
        fuse = self.bn_fuse(fuse)
        fuse = self.relu(fuse)

        # block 4
        #x = self.layer4(fuse3)
        #depth = self.layer4_d(depth)
        #fuse4 = x + depth
        return fuse

    # def forward_downsample(self, rgb, depth):
    #
    #     x = self.conv1(rgb)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #
    #     # block 1
    #     x = self.layer1(x)
    #     # block 2
    #     x = self.layer2(x)
    #     # block 3
    #     x = self.layer3(x)
    #     # block 4
    #     #x = self.layer4(fuse3)
    #     #depth = self.layer4_d(depth)
    #     #fuse4 = x + depth
    #
    #     return x

    def forward(self, rgb_and_depth):
        rgb = rgb_and_depth[0]
        depth = rgb_and_depth[1]

        # print ("INPUT FOR REDNET:")
        # img = rgb.squeeze()
        # img = torchvision.transforms.ToPILImage()(img.cpu())
        # img.save("/home/q/kashapov/maskrcnn-benchmark/input_rednet","PNG")

        # print (rgb.shape)
        # print(depth.shape)

        out = self.forward_downsample(rgb, depth)

        # print ("OUTPUT FROM REDNET: ")
        # print (out.shape)

        return [out]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

