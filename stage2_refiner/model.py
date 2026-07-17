"""Geometry-guided NAFNet that predicts a bounded RGB residual."""

import torch
from torch import nn
import torch.nn.functional as F

from .nafnet import make_blocks


class GeometryGuidedNAFNet(nn.Module):
    def __init__(self, rgb_channels=3, geometry_channels=5, width=64, middle_blocks=8,
                 encoder_blocks=(2, 2, 4, 8), decoder_blocks=(2, 2, 2, 2),
                 max_residual=0.15, modulation_scale=0.1, confidence_gate=False,
                 geometry_mode="full"):
        super().__init__()
        if len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("encoder_blocks and decoder_blocks must have equal length")
        self.rgb_channels = rgb_channels
        self.geometry_channels = geometry_channels
        self.max_residual = float(max_residual)
        self.modulation_scale = float(modulation_scale)
        self.confidence_gate = bool(confidence_gate)
        self.geometry_mode = geometry_mode
        self.padder_size = 2 ** len(encoder_blocks)

        self.rgb_intro = nn.Conv2d(rgb_channels, width, 3, padding=1)
        self.geometry_intro = nn.Conv2d(geometry_channels, width, 3, padding=1)
        self.geometry_modulation = nn.Conv2d(width, width * 2, 3, padding=1)
        self.fusion = nn.Conv2d(width * 2, width, 1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for count in encoder_blocks:
            self.encoders.append(make_blocks(channels, count))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = make_blocks(channels, middle_blocks)
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for count in decoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(make_blocks(channels, count))

        output_channels = 4 if confidence_gate else 3
        self.output_head = nn.Conv2d(width, output_channels, 3, padding=1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        nn.init.zeros_(self.geometry_modulation.weight)
        nn.init.zeros_(self.geometry_modulation.bias)
        if confidence_gate:
            with torch.no_grad():
                self.output_head.bias[3] = -2.0

    def _geometry_mask(self, geometry):
        if self.geometry_mode == "none":
            return torch.zeros_like(geometry)
        if self.geometry_mode == "depth":
            out = torch.zeros_like(geometry)
            out[:, :1] = geometry[:, :1]
            return out
        if self.geometry_mode != "full":
            raise ValueError(f"Unknown geometry_mode: {self.geometry_mode}")
        return geometry

    def _pad(self, tensor):
        height, width = tensor.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    def forward(self, x=None, *, rgb=None, depth=None, normal=None, alpha=None,
                geometry=None, return_residual=False, return_gate=False):
        if x is not None:
            if x.shape[1] != self.rgb_channels + self.geometry_channels:
                raise ValueError(f"Expected {self.rgb_channels + self.geometry_channels} channels, got {x.shape[1]}")
            rgb, geometry = x[:, :self.rgb_channels], x[:, self.rgb_channels:]
        elif geometry is None:
            if any(value is None for value in (rgb, depth, normal, alpha)):
                raise ValueError("Provide x=[RGB,depth,normal,alpha] or all named tensors")
            geometry = torch.cat((depth, normal, alpha), dim=1)
        original_h, original_w = rgb.shape[-2:]
        rgb_padded = self._pad(rgb)
        geometry = self._geometry_mask(self._pad(geometry))

        rgb_feature = self.rgb_intro(rgb_padded)
        geometry_feature = self.geometry_intro(geometry)
        gamma, beta = self.geometry_modulation(geometry_feature).chunk(2, dim=1)
        gamma = self.modulation_scale * torch.tanh(gamma)
        beta = self.modulation_scale * torch.tanh(beta)
        rgb_feature = rgb_feature * (1.0 + gamma) + beta
        features = self.fusion(torch.cat((rgb_feature, geometry_feature), dim=1))

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)

        raw = self.output_head(features)[..., :original_h, :original_w]
        residual = self.max_residual * torch.tanh(raw[:, :3])
        gate = torch.sigmoid(raw[:, 3:4]) if self.confidence_gate else torch.ones_like(residual[:, :1])
        gated_residual = gate * residual
        refined = torch.clamp(rgb + gated_residual, 0.0, 1.0)
        outputs = [refined]
        if return_residual:
            outputs.append(gated_residual)
        if return_gate:
            outputs.append(gate)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
