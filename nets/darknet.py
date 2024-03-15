import math
from collections import OrderedDict

import torch.nn as nn
import time
import torch


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0])
        self.relu1  = nn.LeakyReLU(0.1)
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x):
        # print("残差结构为一个整体")
        print("#res-residual")
        # start1 = time.time()
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        
        # over1 = time.time()
        # print("delay-res",round(over1-start1,5))
        # print("data-res",torch.numel(x) * x.element_size())
        print()
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32


        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)



        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu1  = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        print("这个才是真正的开始吧！！！！！")
        
        print("#1-conv1")
        start1 = time.time()
        x = self.conv1(x)
        over1 = time.time()
        print("delay1",round(over1-start1,5))
        print("data1",torch.numel(x) * x.element_size())
        print()

        print("#2-bn1")
        start1 = time.time()
        x = self.bn1(x)
        over1 = time.time()
        print("delay2",round(over1-start1,5))
        print("data2",torch.numel(x) * x.element_size())
        print()

        print("#3-relu1")
        start1 = time.time()
        x = self.relu1(x)
        over1 = time.time()
        print("delay3",round(over1-start1,5))
        print("data3",torch.numel(x) * x.element_size())
        print()

        print("#4-layer1")
        start1 = time.time()
        x = self.layer1(x)
        over1 = time.time()
        print("delay4",round(over1-start1,5))
        print("data4",torch.numel(x) * x.element_size())
        print()

        print("#5-layer2")
        start1 = time.time()
        x = self.layer2(x)
        over1 = time.time()
        print("delay5",round(over1-start1,5))
        print("data5",torch.numel(x) * x.element_size())
        print()

        print("#6-layer3")
        start1 = time.time()
        out3 = self.layer3(x)
        over1 = time.time()
        print("delay6",round(over1-start1,5))
        print("data6",torch.numel(x) * x.element_size())
        print("#6")
        print()

        print("#7-layer4")
        start1 = time.time()
        out4 = self.layer4(out3)
        over1 = time.time()
        print("delay7",round(over1-start1,5))
        print("data7",torch.numel(x) * x.element_size())
        print("#7")
        print()

        print("#8-layer5")
        start1 = time.time()
        out5 = self.layer5(out4)
        over1 = time.time()
        print("delay8",round(over1-start1,5))
        print("data8",torch.numel(x) * x.element_size())
        print("#8")
        print()
        print("darkNet 结束")


        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model
