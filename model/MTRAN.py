import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ECA import ECA
from model.ir50 import get_ir50, load_pretrained_weights

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        ir_50= get_ir50(50, 0.0, 'ir')
        ir_checkpoint = torch.load(r'./pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        ir_50 = load_pretrained_weights(ir_50, ir_checkpoint)
        self.backbone = ir_50

    def forward(self, x):
        x = self.backbone(x)[2]
        return x

def FRFD(x):
    B, C, H, W = x.size()
    h_half, w_half = H // 2, W // 2
    feat_tl = x[:, :, :h_half, :w_half]
    feat_tr = x[:, :, :h_half, w_half:]
    feat_bl = x[:, :, h_half:, :w_half]
    feat_br = x[:, :, h_half:, w_half:]
    feats = [feat_tl, feat_tr, feat_bl, feat_br]
    return feats

class ScaledDotProductAttention(nn.Module):
    def __init__(self, channels=256):
        super(ScaledDotProductAttention, self).__init__()

        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = math.sqrt(channels)

    def forward(self, query, key, value):
        B, C, H, W = query.shape

        q = self.q_conv(query).view(B, C, -1)  # [B, 256, 49]
        k = self.k_conv(key).view(B, C, -1)  # [B, 256, 49]
        v = self.v_conv(value).view(B, C, -1)  # [B, 256, 49]

        # [B, 49, 256]
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, 49, 49]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # [B, 49, 256]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, 256, 7, 7]
        return out

class MTRA(nn.Module):
    def __init__(self, channels=256):
        super(MTRA, self).__init__()
        self.mtra = nn.ModuleList()
        for i in range(4):
            mtra1 = nn.ModuleList()
            for j in range(4):
                mtra2 = nn.ModuleList()
                for k in range(4):
                    mtra2.append(ScaledDotProductAttention(channels))
                mtra1.append(mtra2)
            self.mtra.append(mtra1)
    def forward(self, feats):
        atten_feats = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    feat = self.mtra[i][j][k](feats[i], feats[j], feats[k])
                    atten_feats.append(feat)
        concat_feats = torch.cat(atten_feats, dim=1)
        return concat_feats

class Classifier(nn.Module):
    def __init__(self, channels=256, num_classes=7):
        super(Classifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(channels * 64, channels * 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(channels * 32, channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(channels, num_classes),
        )
    def forward(self, fused_feat):
        x = self.pool(fused_feat).view(-1, 256 * 64)
        x = self.linear(x)
        return x

class MTRAN(nn.Module):
    def __init__(self, num_classes=7):
        super(MTRAN, self).__init__()
        channels = 256
        self.backbone = Backbone()
        self.mtra = MTRA(channels)
        self.eca = ECA()
        self.classifier = Classifier(channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        feats = FRFD(x)
        concat_feats = self.mtra(feats)
        fused_feat = self.eca(concat_feats)
        out = self.classifier(fused_feat)
        return out


if __name__ == "__main__":
    model = MTRAN(num_classes=7)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    x = torch.randn(16, 3, 224, 224)
    y = model(x)
    print("The shape of output:", y.shape)
