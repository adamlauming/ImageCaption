import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.init_weights import init_weights

#===============================================================================
# SCM
#===============================================================================
class SCM(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(SCM, self).__init__()
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.conv_v_1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k_1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q_1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_v_2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k_2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q_2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

    def forward(self, x_1, x_2):
        V_1 = self.conv_v_1(x_1)
        K_1 = self.conv_k_1(x_1)
        Q_1 = self.conv_q_1(x_1)
        shape = V_1.shape
        # print(shape)
        V_1 = V_1.view(V_1.shape[0], V_1.shape[1], -1)
        K_1 = K_1.view(K_1.shape[0], K_1.shape[1], -1)
        Q_1 = Q_1.view(Q_1.shape[0], Q_1.shape[1], -1)

        V_2 = self.conv_v_2(x_2)
        K_2 = self.conv_k_2(x_2)
        Q_2 = self.conv_q_2(x_2)    
        V_2 = V_2.view(V_2.shape[0], V_2.shape[1], -1)
        K_2 = K_2.view(K_2.shape[0], K_2.shape[1], -1)
        Q_2 = Q_2.view(Q_2.shape[0], Q_2.shape[1], -1)

        R12 = torch.sigmoid(torch.matmul(torch.transpose(Q_1, -1, -2), K_2))
        R21 = torch.sigmoid(torch.matmul(torch.transpose(Q_2, -1, -2), K_1))
        # print(R12.shape)

        Fu_1 = torch.matmul(R12, torch.transpose(V_2, -1, -2))
        Fu_2 = torch.matmul(R21, torch.transpose(V_1, -1, -2))
        Fu_1 = torch.transpose(Fu_1, -1, -2).view(shape)
        Fu_2 = torch.transpose(Fu_2, -1, -2).view(shape)
        # print(Fu_1.shape)

        y_1 = torch.cat([x_1, Fu_1], dim=1)
        y_2 = torch.cat([x_2, Fu_2], dim=1)
        # print(y_1.shape)
        return y_1, y_2


#===============================================================================
# PSP module
#===============================================================================
def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

#===============================================================================
# EADAM
#===============================================================================
class EADAM_V1(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.conv_v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

    def forward(self, x_en, x_de):
        V = self.conv_v(x_en)
        K = self.conv_k(x_en)
        Q = self.conv_q(x_de)
        shape = V.shape

        V = V.view(V.shape[0], V.shape[1], -1)
        K = K.view(V.shape[0], K.shape[1], -1)
        Q = Q.view(Q.shape[0], Q.shape[1], -1)

        H = torch.matmul(V, torch.transpose(K, -1, -2))
        H = torch.matmul(torch.sigmoid(H), Q) / (V.shape[1] * V.shape[2])**0.5

        H = H.view(shape)
        attmap = self.conv_out(H)

        return attmap + x_en + x_de


#===============================================================================
# multiscale input Module
#===============================================================================
class MultiInput(nn.Module):
    def __init__(self, in_channels, factor):
        super().__init__()
        self.conv_in = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_out = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.factor = factor

    def forward(self, img, features):
        x = F.interpolate(img, scale_factor=self.factor, mode='bilinear', align_corners=True)
        x = self.conv_in(x)
        x = self.bn(x)
        out = torch.cat([x, features], dim=1)
        out = self.conv_out(out)

        return out


#===============================================================================
# TopHat Module
#===============================================================================
def dilation(x, size):
    maxpool = nn.MaxPool2d((size, size), stride=(1, 1), padding=size // 2)

    return maxpool(x)


def erosion(x, size):
    maxpool = nn.MaxPool2d((size, size), stride=(1, 1), padding=size // 2)

    return -maxpool(-x)


def black_tophat(x, size):
    x_dilate = dilation(x, size)
    x_close = erosion(x_dilate, size)

    return x_close - x


class TopHatBlockV2(nn.Module):
    def __init__(self, in_channels, poolsize=[3, 5, 7]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)

        self.poolsize = poolsize
        self.conv_1x1 = nn.Conv2d(in_channels * len(poolsize), in_channels, 1)

    def forward(self, x):
        identity = x

        x = self.conv_in(x)

        out = []
        for idx, size in enumerate(self.poolsize):
            out.append(black_tophat(x, size))

        out = torch.cat(out, dim=1)
        out = self.conv_1x1(out)

        return out + identity


#===============================================================================
# PCA Blocks
#===============================================================================
class SAPCABlockV5(nn.Module):
    """
    Spatial attention block
    """

    def __init__(self, in_channels, scale=3):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        gx = self.conv1(x)
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)
        gx_ = gx_ - torch.mean(gx_, dim=-1, keepdim=True)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -16:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        z = torch.matmul(w, gx_)  # 这一步可以考虑加softmax
        z = F.softmax(z * self.scale, dim=1)
        y_ = torch.matmul(w.transpose(-1, -2), z)
        y_ = y_.view(gx.shape[0], gx.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y_)

        x_en = x + attmap

        return x_en


class SAPCABlockV4(nn.Module):
    """
    Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                   # norm_layer(in_channels),
                                   )

    def forward(self, x):
        gx = self.conv1(x)
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)
        gx_ = gx_ - torch.mean(gx_, dim=-1, keepdim=True)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        z = torch.matmul(w, gx_)  # 这一步可以考虑加softmax
        y_ = torch.matmul(w.transpose(-1, -2), z)
        y_ = y_.view(gx.shape[0], gx.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y_)

        x_en = x + attmap

        return x_en


#================================================================
#================================================================
#================================================================
class SAPCABlockV3(nn.Module):
    """
    # Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        gx = self.conv1(x)
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)
        gx_ = gx_ - torch.mean(gx_, dim=-1, keepdim=True)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        y = torch.matmul(w, gx_)
        y = y.view(y.shape[0], y.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y)
        x_en = x + attmap

        return x_en


class SAPCABlockV2(nn.Module):
    """
    #Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        gx = self.prelu(self.conv1(x))
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        y = torch.matmul(w, gx_)
        y = torch.sigmoid(y)
        y = y.view(y.shape[0], y.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y)
        x_en = x * torch.tanh(attmap)

        return x_en


class SAPCABlock(nn.Module):
    """
    #Spatial attention block
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        gx = self.prelu(self.conv1(x))
        gx_ = gx.view(gx.shape[0], gx.shape[1], -1)

        cmat = torch.matmul(gx_, gx_.transpose(-1, -2)) / gx.shape[0]
        w = []
        for i in range(cmat.shape[0]):
            eigval, eigvec = torch.symeig(cmat[i], eigenvectors=True)
            w.append(eigvec[:, -x.shape[1] // 4:].transpose(-1, -2))
        w = torch.stack(w, dim=0)

        y = torch.matmul(w, gx_)
        y = y.view(y.shape[0], y.shape[1], gx.shape[2], gx.shape[3])

        attmap = self.conv2(y)
        x_en = x + attmap

        return x_en


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x
