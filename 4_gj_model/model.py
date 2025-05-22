import torch
import torch.nn as nn
import torch.nn.functional as F

class UltraLightNet(nn.Module):
    def __init__(self, init_weights=False):
        super(UltraLightNet, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.features(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CombinedNet(nn.Module):
    def __init__(self, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., num_classes=10):
        super(CombinedNet, self).__init__()
        self.cnn = UltraLightNet()
        self.attention = nn.ModuleList([
            Attention(dim=256, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio)
            for _ in range(2)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 14 * 14, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for attn_layer in self.attention:
            x = attn_layer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = x.reshape(B, -1)
        x = self.classifier(x)
        return x
