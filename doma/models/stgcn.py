import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x
    
class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(SpatioConvLayer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk.to(x.device), x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b.to(x.device)
        return torch.relu(x_gc + x)

class STBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(STBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], Lk)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return self.dropout(x_ln)

# class FullyConvLayer(nn.Module):
#     def __init__(self, c):
#         super(FullyConvLayer, self).__init__()
#         self.conv = nn.Conv2d(c, 1, 1)

#     def forward(self, x):
#         return self.conv(x)

# class OutputLayer(nn.Module):
#     def __init__(self, c, T, n):
#         super(OutputLayer, self).__init__()
#         self.tconv1 = TemporalConvLayer(T, c, c, "GLU")
#         self.ln = nn.LayerNorm([n, c])
#         self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")
#         self.fc = FullyConvLayer(c)

#     def forward(self, x):
#         x_t1 = self.tconv1(x)
#         x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         x_t2 = self.tconv2(x_ln)
#         return self.fc(x_t2)

class STGCN(nn.Module):
    def __init__(self, ks, kt, bs, T, n, Lk, p, num_classes):
        super(STGCN, self).__init__()
        self.st_conv1 = STBlock(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = STBlock(ks, kt, n, bs[1], p, Lk)

        final_c = bs[1][2]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_c, num_classes)

    def forward(self, x):
        x = self.st_conv1(x)
        x = self.st_conv2(x)    # (B, C, 1, 1)

        x = self.pool(x).squeeze(-1).squeeze(-1) # (B, C)
        out = self.classifier(x) # (B, num_classes)
        return out

