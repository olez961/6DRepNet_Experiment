import math

import torch
from torch import nn

from backbone.repvgg import get_RepVGG_func_by_name
from backbone import convnext
import utils

class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6),
                 droBatchNorm=nn.BatchNorm2d,
                 pretrained=True, 
                 gpu_id=0):
        super(SixDRepNet, self).__init__()
        self.gpu_id = gpu_id
        # 下面做出的一些改变是我为了更改backbone但是又需要保留原backbone的一些不得已的操作
        # 希望问题不大
        self.repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        if self.repvgg_fn:
            self.backbone = self.repvgg_fn(deploy)
            if pretrained:
                checkpoint = torch.load(backbone_file)
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                ckpt = {k.replace('module.', ''): v for k,
                        v in checkpoint.items()}  # strip the names
                self.backbone.load_state_dict(ckpt)
        else:
            self.backbone = convnext.convnext_small(pretrained=True)

        if self.repvgg_fn:
            self.layer0 = self.backbone.stage0
            self.layer1 = self.backbone.stage1
            self.layer2 = self.backbone.stage2
            self.layer3 = self.backbone.stage3
            self.layer4 = self.backbone.stage4
        else:
            self.layer0 = None
            self.layer1 = None
            self.layer2 = None
            self.layer3 = None
            self.layer4 = self.backbone.stages[-1]
        # 全局平均池化层gap用于将特征的尺寸降到1x1
        # 把nn.AdaptiveAvgPool2d的输出尺寸设置为1，意味着对每一个输入特征图，会产生一个元素的输出。
        # 把整个特征图投影到一个点上，并对整个特征图求得其平均值。
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        # 遍历网络的第4个部分（self.layer4）中的每个模块，
        # 并通过判断该模块是否是一个卷积层并且模块名称包含"rbr_dense"或"rbr_reparam"。
        # 如果是这样的模块，它会记录该卷积层的输出通道数作为最后一个通道。
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        # fea_dim可能是feature dimension的缩写
        if self.repvgg_fn:
            fea_dim = last_channel
        else:
            fea_dim = 768
        # 线性层linear_reg将特征的尺寸映射到6维
        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        if self.repvgg_fn:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            for i in range(4):
                x = self.backbone.downsample_layers[i](x)
                x = self.backbone.stages[i](x)
        x= self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        if self.gpu_id ==-1:
            return utils.compute_rotation_matrix_from_ortho6d(x, False, self.gpu_id)
        else:
            return utils.compute_rotation_matrix_from_ortho6d(x, True, self.gpu_id)





class SixDRepNet2(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      


        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = utils.compute_rotation_matrix_from_ortho6d(x)

        return out