import torchvision
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch
import math
import numpy as np
import copy
import itertools
import torch.nn.functional as F
from torch.nn import init
"""
x1和x2,x_mix一起输入,x1和x2进行local classification, loss值对应entropy,每个cell, loss越小,交叉熵越小, 能代表target的可能性越大.mix时候取到的概率更大
"""
"""
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
"""
################################

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],**kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet18'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    ##load model trained from imagenet
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)

    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    ##load model trained from imagenet
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet101'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)

    return model

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Returns 2D convolutional layer with space-preserving padding
"""https://github.com/amiltonwong/pytorch_fcn/blob/master/model.py"""
def conv(in_planes, out_planes, kernel_size=3, stride=1,  padding=1, output_padding=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, bias=bias)

    # Bilinear interpolation init
    w = torch.Tensor(kernel_size, kernel_size)
    # print(layer.weight.shape, w.div(in_planes).repeat(out_planes, in_planes, 1, 1).shape)
    # torch.Size([2048, 1024, 3, 3])
    # torch.Size([1024, 2048, 3, 3])
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  # if bias:
  #   init.constant(layer.bias, 0)
  return layer



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 网络输入部分
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1_3k3 = nn.Conv2d(3, 64, kernel_size=3,padding=1, bias=False)#修改点1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 中间卷积部分
        # self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer1_3k3 = self._make_layer(block, 64, layers[0], stride=1)#修改点2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.conv5 = conv(512 * block.expansion, 128 * block.expansion, stride=2, transposed=True)
        self.bn5 = nn.BatchNorm2d(128 * block.expansion)
        self.conv6 = conv(128 * block.expansion + 256 * block.expansion, 64 * block.expansion, stride=2, transposed=True)
        self.bn6 = nn.BatchNorm2d(64 * block.expansion)
        self.conv7 = conv(64 * block.expansion + 128 * block.expansion, 32 * block.expansion, stride=2, transposed=True)
        self.bn7 = nn.BatchNorm2d(32 * block.expansion)
        self.conv8 = conv(32 * block.expansion + 64 * block.expansion, 64, kernel_size=1, stride=1, padding=0, output_padding=0, transposed=True)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = conv(64 + 64, 64, kernel_size=1, stride=1, padding=0, output_padding=0, transposed=True)
        self.bn9 = nn.BatchNorm2d(64)
        # self.conv10 = conv(32, num_classes, kernel_size=7)
        # init.constant(self.conv10.weight, 0)  # Zero init

        self.SA = SelfAttention(input_size=64)
        self.SAP = self.SuperpixelAttentionPooling

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc2 = nn.Linear(2112, num_classes)
        self.fc_local = nn.Linear(16 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):#是用来构建ResNet网络中的4个blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        #将每个blocks的第一个residual结构保存在layers列表中
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #将每个blocks的剩下residual 结构保存在layers列表中
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def SuperpixelAttentionPooling(self, x, SuperP_mask, atten_top_ratio):
        # print(x.shape)#torch.Size([32, 256, 32, 32])
        avgpool_sa_batch_sel = []
        avgpool_sa_batch = []
        topN_idx_batch = []
        for sp in range(x.shape[0]):
            mask_value = np.unique(SuperP_mask[sp])
            x_sp = x[sp].reshape(x.shape[2], x.shape[3], x.shape[1])

            avgpool = []
            for v in mask_value:
                # print(x_sp.shape)
                # print(x_sp[SuperP_mask[sp]==v].shape)
                # print(x_sp[SuperP_mask[sp] == v].mean(0).shape)
                avgpool.append(x_sp[SuperP_mask[sp]==v].mean(0))

            avgpool = torch.stack(avgpool)

            avgpool_sa = self.SA(avgpool)

            avgpool_sa_spacial = avgpool_sa.sum(1)
            _, map_topN_idx = torch.topk(avgpool_sa_spacial, int(atten_top_ratio*(avgpool_sa_spacial.shape[0])), dim=0, largest=True)

            topN_idx_batch.append(map_topN_idx)
            avgpool_sa_batch.append(avgpool_sa)

            if len(map_topN_idx)>0:
                avgpool_sa_sp = [avgpool_sa[idx] for idx in map_topN_idx]
                avgpool_sa_sp = torch.stack(avgpool_sa_sp)
                avgpool_sa_batch_sel.append(avgpool_sa_sp)
            else:
                avgpool_sa_batch_sel.append(avgpool_sa)

        return avgpool_sa_batch, avgpool_sa_batch_sel, topN_idx_batch

    def forward(self, x, local=False, superpixel_map=None, topN_local_ratio=None):
        x0 = self.relu(self.bn1(self.conv1_3k3(x)))
        # print(x0.shape) #torch.Size([32, 64, 32, 32])
        # x = self.maxpool(x)
        x1 = self.layer1_3k3(x0)
        # print(x1.shape)  #torch.Size([32, 256, 32, 32])
        x2 = self.layer2(x1)
        # print(x2.shape) #torch.Size([32, 512, 16, 16])
        x3 = self.layer3(x2)
        # print(x3.shape)  #torch.Size([32, 1024, 8, 8])
        x4 = self.layer4(x3)
        # print(x4.shape)   #torch.Size([32, 2048, 4, 4])


        x_up1 = self.relu(self.bn5(self.conv5(x4)))
        # print(x_up1.shape, x3.shape)  #torch.Size([32, 512, 8, 8]) torch.Size([32, 1024, 8, 8])
        # print(torch.cat((x_up1, x3), dim=1).shape) #torch.Size([32, 1536, 8, 8])
        x_up2 = self.relu(self.bn6(self.conv6(torch.cat((x_up1, x3), dim=1))))
        # print(x_up2.shape, x2.shape)  #torch.Size([32, 256, 16, 16]) torch.Size([32, 512, 16, 16])
        # print(torch.cat((x_up2, x2), dim=1).shape)  # torch.Size([32, 1536, 8, 8]) #torch.Size([32, 768, 16, 16])
        x_up3 = self.relu(self.bn7(self.conv7(torch.cat((x_up2, x2), dim=1))))
        # print(x_up3.shape, x1.shape)  #torch.Size([32, 256, 32, 32]) torch.Size([32, 256, 32, 32])
        x_up4 = self.relu(self.bn8(self.conv8(torch.cat((x_up3, x1), dim=1))))
        # print(x_up4.shape, x0.shape)  #torch.Size([32, 64, 32, 32]) torch.Size([32, 64, 32, 32])
        x_up5 = self.bn9(self.conv9(torch.cat((x_up4, x0), dim=1)))

        """z^ classification"""
        pre_logit_Z = self.avgpool(x4)
        pre_logit_Zhat = self.avgpool(x_up5)
        pre_logit = torch.cat((pre_logit_Z, pre_logit_Zhat), dim=1)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits2 = self.fc2(pre_logit)

        if local:
            x_sap_list, x_sap_sel_list, topN_local_idx = self.SAP(x_up5, superpixel_map, topN_local_ratio)  # list(32*[n, 256])
            x_locals_out_batch = []
            weights_locals_out_batch = []
            for sp in range(len(x_sap_sel_list)):
                x_locals_out_sp = []
                for tk in range(x_sap_sel_list[sp].shape[0]):
                    x_local_out = self.fc_local(x_sap_sel_list[sp][tk])
                    x_locals_out_sp.append(x_local_out)
                x_locals_out_sp = torch.stack(x_locals_out_sp)
                x_locals_out_batch.append(x_locals_out_sp)
                weights_locals_out_batch.append(F.sigmoid(x_sap_list[sp].sum(1)))
                # print(F.sigmoid(x_sap_list[sp].sum(1)))
            return logits2, x_locals_out_batch, weights_locals_out_batch, topN_local_idx
        else:
            return logits2

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample   #对输入特征图大小进行减半处理
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


"""https://blog.csdn.net/beilizhang/article/details/115282604"""


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention,self).__init__()

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        # self.out_dropout = nn.Dropout(dropout_prob)
        self.hidden_size = input_size
        self.LayerNorm = LayerNorm(input_size, eps=1e-12)


    def forward(self, input_tensor):
        """input tensor (n,d)"""
        query_layer = self.query(input_tensor)
        key_layer = self.key(input_tensor)
        value_layer = self.value(input_tensor)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.out_dropout(attention_probs)

        hidden_states = torch.matmul(attention_probs, value_layer)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

