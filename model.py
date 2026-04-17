import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
#from inception import ECB
import numpy as np
import kornia
import math


class DWTForward(nn.Module):
    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y

class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_theta = nn.Conv2d(9, 64, 3, 1, 1, bias=True)
        self.theta_1 = nn.Conv2d(64,64,1)
        self.theta_2 = nn.Conv2d(64,64,1)
        self.theta_3 = nn.Conv2d(64,1,1)
        self.relu = nn.ReLU(inplace=False)
        self.sigmod = nn.Sigmoid()
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        #self.theta = theta

    def theta_net(self, high_freq):
        avg_high = self.relu(self.conv_theta(high_freq))
        avg_high = self.average_pool(avg_high)
        x = self.relu(self.theta_1(avg_high))
        x = self.relu(self.theta_2(x))
        theta = self.sigmod(self.theta_3(x))
        return theta

    def forward(self, x , high_freq):
        theta = self.theta_net(high_freq)
        theta = theta.reshape(-1)
        out_normal = self.conv(x)

        if math.fabs(theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - theta * out_diff


class attention_enhance_light_rgb(nn.Module):

    def __init__(self,basic_conv = Conv2d_cd):
        super(attention_enhance_light_rgb, self).__init__()

        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.sigmod = nn.Sigmoid()
        self.tanh = nn.Tanh()
        number_f = 32
        self.FC_1 = nn.Conv2d(1,number_f,1)
        self.FC_2 = nn.Conv2d(number_f,1,1)

        self.encoder_org_1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.encoder_org_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_org_3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_org_4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_org_5 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        self.encoder_down2_1 = nn.Conv2d(2, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)

        self.encoder_down4_1 = nn.Conv2d(2, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)


        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 2, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.average_pool = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim=0)
        self.fc1   = nn.Linear(64, 512) # 全连接
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)  # out: number_channel

        self.to_gamma_0 = nn.Sequential(nn.Linear(512, 32),nn.Sigmoid())
        self.to_beta_0 = nn.Sequential(nn.Linear(512, 32), nn.Tanh())

        self.res_conv = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.decoder_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.decoder_2 = nn.Conv2d(number_f, 1, 3, 1, 1, bias=True)

        self.low_freq = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.cdc = basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.dwt_trans = DWTForward().cuda()

    def cal_batch_histogram_torch(self, v_channel):
        histogram_list = torch.empty(0).cuda()
        for i in range(np.shape(v_channel)[0]):
            v = v_channel[i, :, :, :]
            v = v.squeeze(0) * 255
            histogram = torch.histc(v, 256, min=0, max=255)
            histogram = histogram.unsqueeze(0)
            histogram_list = torch.cat((histogram_list, histogram), 0)

        return histogram_list

    def cal_batch_nograd(self,v_channel):
        v_hist = torch.zeros(0).cuda()
        for i in range(np.shape(v_channel)[0]):
            h_hist = torch.histc(v_channel[i] * 255, 256, min=0, max=255).detach().unsqueeze(0)
            v_hist = torch.cat((v_hist, h_hist), 0)
        return v_hist

    def attention_net(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        map = self.sigmod(self.fc3(x))

        return map

    def statistic_net(self,x):

        hist_x = self.cal_batch_histogram_torch(x)

        return hist_x

    def embed_net(self, x , gamma , beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res_conv(x) + beta
        return x + res

    def forward(self, x):

        f_org_0 = self.relu(self.encoder_org_1(x))
        f_org_1 = self.relu(self.encoder_org_2(f_org_0))
        f_org_2 = self.relu(self.encoder_org_3(f_org_1))
        f_org_3 = self.relu(self.encoder_org_4(f_org_2))
        f_org_4 = self.relu(self.encoder_org_5(f_org_3))

        #high_freq , low_freq

        dwt_lowimg = self.dwt_trans(x)

        c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12 = torch.split(dwt_lowimg, 1, dim=1)
        haar_A = torch.cat((c_1, c_5, c_9), dim=1)
        haar_H = torch.cat((c_2, c_6, c_10), dim=1)
        haar_V = torch.cat((c_3, c_7, c_11), dim=1)
        haar_D = torch.cat((c_4, c_8, c_12), dim=1)
        low_freq = haar_A
        high_freq = torch.cat((haar_H,haar_V,haar_D),1)


        f_deco = self.relu(self.cdc(f_org_4,high_freq))

        attention_map = self.relu(self.low_freq(low_freq))
        attention_map = self.average_pool(attention_map)
        print(np.shape(attention_map))
        attention_map = self.attention_net(attention_map)
        gamma_0 = self.to_gamma_0(attention_map)
        beta_0 = self.to_beta_0(attention_map)

        f = self.embed_net(f_deco, gamma_0, beta_0)
        f = self.relu(self.decoder_1(f))
        out_map = self.tanh(self.decoder_2(f))


        enhance_image = x + out_map * (torch.pow(x, 2) - x)

        return enhance_image,out_map


class attention_enhance_light(nn.Module):

    def __init__(self):
        super(attention_enhance_light, self).__init__()

        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.sigmod = nn.Sigmoid()
        self.tanh = nn.Tanh()
        number_f = 32
        self.FC_1 = nn.Conv2d(1,number_f,1)
        self.FC_2 = nn.Conv2d(number_f,1,1)

        self.encoder_org_1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.encoder_org_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_org_3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_org_4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_org_5 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        self.encoder_down2_1 = nn.Conv2d(2, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down2_5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)

        self.encoder_down4_1 = nn.Conv2d(2, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.encoder_down4_5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)


        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 2, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.average_pool = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim=0)
        self.fc1   = nn.Linear(256, 512) # 全连接
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)  # out: number_channel

        self.to_gamma_0 = nn.Sequential(nn.Linear(512, 32),nn.Sigmoid())
        self.to_beta_0 = nn.Sequential(nn.Linear(512, 32), nn.Tanh())

        self.res_conv = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.decoder_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.decoder_2 = nn.Conv2d(number_f, 1, 3, 1, 1, bias=True)



    def cal_batch_histogram_torch(self, v_channel):
        histogram_list = torch.empty(0).cuda()
        for i in range(np.shape(v_channel)[0]):
            v = v_channel[i, :, :, :]
            v = v.squeeze(0) * 255
            histogram = torch.histc(v, 256, min=0, max=255)
            histogram = histogram.unsqueeze(0)
            histogram_list = torch.cat((histogram_list, histogram), 0)

        return histogram_list

    def cal_batch_nograd(self,v_channel):
        v_hist = torch.zeros(0).cuda()
        for i in range(np.shape(v_channel)[0]):
            h_hist = torch.histc(v_channel[i] * 255, 256, min=0, max=255).detach().unsqueeze(0)
            v_hist = torch.cat((v_hist, h_hist), 0)
        return v_hist

    def attention_net(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        map = self.sigmod(self.fc3(x))

        return map

    def statistic_net(self,x):

        hist_x = self.cal_batch_histogram_torch(x)

        return hist_x

    def embed_net(self, x , gamma , beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma)*self.res_conv(x) + beta
        return x + res

    def forward(self, x):

        f_org_0 = self.relu(self.encoder_org_1(x))
        f_org_1 = self.relu(self.encoder_org_2(f_org_0))
        f_org_2 = self.relu(self.encoder_org_3(f_org_1))
        f_org_3 = self.relu(self.encoder_org_4(f_org_2))
        f_org_4 = self.relu(self.encoder_org_5(f_org_3))

        statistic_map = self.statistic_net(x)
        statistic_map = statistic_map.detach()
        attention_map = self.attention_net(statistic_map)
        gamma_0 = self.to_gamma_0(attention_map)
        beta_0 = self.to_beta_0(attention_map)

        f = self.embed_net(f_org_4,gamma_0,beta_0)
        f = self.relu(self.decoder_1(f))
        out_map = self.tanh(self.decoder_2(f))

        enhance_image = x + out_map * (torch.pow(x, 2) - x)

        return enhance_image,out_map

class light_class(nn.Module):
    def __init__(self, in_dim=256, hidden1=128, hidden2=64, hidden3=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x should be histogram
        # support [B,256], [B,1,256], [B,256,1,1]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.squeeze(1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported BPNet input shape: {x.shape}")

        if x.size(1) != 256:
            raise ValueError(f"BPNet expects 256-dim histogram, got shape {x.shape}")

        return self.mlp(x).squeeze(1)