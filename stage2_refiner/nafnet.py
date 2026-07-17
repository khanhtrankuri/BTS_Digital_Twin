"""Small, dependency-free NAFNet building blocks."""

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)
        ctx.eps = eps
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        y, var, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, -1, 1, 1)
        mean_grad = grad.mean(1, keepdim=True)
        mean_grad_y = (grad * y).mean(1, keepdim=True)
        grad_x = (grad - mean_grad - y * mean_grad_y) / torch.sqrt(var + ctx.eps)
        grad_weight = (grad_output * y).sum((0, 2, 3))
        grad_bias = grad_output.sum((0, 2, 3))
        return grad_x, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNorm2dFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    def __init__(self, channels, depthwise_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw_channels = channels * depthwise_expand
        ffn_channels = channels * ffn_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.gate1 = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channels // 2, dw_channels // 2, 1))
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.gate2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.conv3(self.sca(self.gate1(self.conv2(self.conv1(self.norm1(x))))))
        x = x + self.dropout1(y) * self.beta
        y = self.conv5(self.gate2(self.conv4(self.norm2(x))))
        return x + self.dropout2(y) * self.gamma


def make_blocks(channels, count):
    return nn.Sequential(*[NAFBlock(channels) for _ in range(int(count))])
