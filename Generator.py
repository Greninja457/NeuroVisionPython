import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_ch, base),
            ConvBlock(base, base * 2, stride=2),
            ConvBlock(base * 2, base * 4, stride=2)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch // 2, in_ch // 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, num_res_blocks=6):
        super().__init__()

        self.low_encoder = Encoder()
        self.ref_encoder = Encoder()

        self.fusion_conv = ConvBlock(512, 256)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_res_blocks)]
        )

        self.decoder = Decoder(256)

    def forward(self, low_img, ref_imgs):
        """
        low_img: [B,3,H,W]
        ref_imgs: list of 3 tensors
        """

        low_feat = self.low_encoder(low_img)

        ref_feats = [self.ref_encoder(r) for r in ref_imgs]
        ref_feat = torch.mean(torch.stack(ref_feats), dim=0)

        fused = torch.cat([low_feat, ref_feat], dim=1)
        fused = self.fusion_conv(fused)

        x = self.res_blocks(fused)
        x = self.decoder(x)

        return torch.clamp(low_img + x, -1, 1)
