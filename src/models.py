"""
Generator / Critic network definitions.

Organized by experiment, since each domain (2D toy data, MNIST, CartoonSet)
uses a different architecture. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 2D toy experiment - small MLPs mapping R^2 -> R^2 / R^2 -> R
# ============================================================================

class Generator(nn.Module):
    """Primal network. Maps a 2D source point to a 2D target-space point."""

    def __init__(self, input_dim=2, h1=16, h2=32):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.bot = nn.Linear(h2, 2)  # output: 2D coordinates
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        return self.bot(out)


class Critic(nn.Module):
    """Dual network (unconstrained). Scores a 2D point."""

    def __init__(self, input_dim=2, h1=32, h2=16):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.bot = nn.Linear(h2, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.leaky_relu(self.layer1(x))
        out = self.leaky_relu(self.layer2(out))
        return self.bot(out)


# ============================================================================
# MNIST experiment -- unconditional. U-Net generator + conv critic, both
# operating on 32x32 grayscale images (Section 4.6 of the thesis).
# ============================================================================

class MNISTCritic(nn.Module):
    """Five-layer conv critic: 32x32 image -> scalar score. Unconstrained
    (no spectral norm / gradient penalty) -- see Section 4.1 of the thesis
    for why the dual variable is deliberately left unbounded."""

    def __init__(self, channels_img=1, features_d=32, img_size=32):
        super().__init__()
        final_kernel = img_size // (2 ** 4)  # = 2 for 32x32 input

        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 8, 1, kernel_size=final_kernel, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class MNISTGenerator(nn.Module):
    """U-Net generator with skip connections. Encoder downsamples 32x32 ->
    1x1 in 5 conv layers; decoder mirrors it with transposed convs, each
    stage concatenating the matching encoder feature map."""

    def __init__(self, channels_img=1, features_g=32):
        super().__init__()
        f = features_g

        # Encoder: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels_img, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(
            nn.Conv2d(f * 8, f * 16, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3))  # bottleneck only

        # Decoder: mirrors the encoder, with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(f * 8 * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(f * 4 * 2, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(f * 2 * 2, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.out = nn.Sequential(
            nn.ConvTranspose2d(f * 2, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)   # [f,    16, 16]
        e2 = self.enc2(e1)  # [2f,    8,  8]
        e3 = self.enc3(e2)  # [4f,    4,  4]
        e4 = self.enc4(e3)  # [8f,    2,  2]
        e5 = self.enc5(e4)  # [16f,   1,  1]  bottleneck

        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        return self.out(torch.cat([d4, e1], dim=1))


# ============================================================================
# MNIST experiment -- class-conditional. Same U-Net/conv-critic shapes as
# above, with a class label injected into both networks (Section 4.6.3).
# ============================================================================

class CondGenerator(nn.Module):
    """U-Net generator with class conditioning via a spatial label channel
    concatenated at the input. The label embedding is reshaped to [B,1,H,W]
    and stacked alongside the noise channel, so label info flows through
    every encoder layer (rather than being injected at the bottleneck)."""

    def __init__(self, channels_img=1, features_g=32, num_classes=10, img_size=32):
        super().__init__()
        f = features_g
        self.f = f
        self.num_classes = num_classes
        self.img_size = img_size

        # Class embedding lives in image space (one scalar per pixel) and is
        # concatenated as an extra channel onto the noise.
        self.class_emb = nn.Embedding(num_classes, img_size * img_size)
        in_ch = channels_img + 1  # noise + class map

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(
            nn.Conv2d(f * 8, f * 16, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3))

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(f * 8 * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(f * 4 * 2, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(f * 2 * 2, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2))
        self.out = nn.Sequential(
            nn.ConvTranspose2d(f * 2, channels_img, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid())

    def _class_map(self, y):
        return self.class_emb(y).view(y.size(0), 1, self.img_size, self.img_size)

    def forward(self, x, y):
        x_cat = torch.cat([x, self._class_map(y)], dim=1)  # [B, 2, H, W]

        e1 = self.enc1(x_cat)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        return self.out(torch.cat([d4, e1], dim=1))


class ProjCritic(nn.Module):
    """

    The base scoring head is structurally identical to `MNISTCritic` (same
    conv stack + same final 2x2 conv). The class signal enters via an
    additive projection term <emb(y), sum_pool(features)>:

        D(x, y) = Critic_final_conv(features(x)) + <emb(y), pool(features(x))>
    """

    def __init__(self, channels_img=1, features_d=32, img_size=32, num_classes=10):
        super().__init__()
        final_kernel = img_size // (2 ** 4)  # = 2 for 32x32 input

        # Conv stack -- identical to the first 8 layers of MNISTCritic.
        self.features = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        # Final 2x2 conv -- identical to the last layer of MNISTCritic.
        self.final_conv = nn.Conv2d(features_d * 8, 1, kernel_size=final_kernel, stride=1, padding=0)

        # Projection branch.
        self.label_emb = nn.Embedding(num_classes, features_d * 8)

    def forward(self, x, y):
        feat = self.features(x)               # [B, fd*8, 2, 2]
        base = self.final_conv(feat)           # [B, 1, 1, 1] -- same as MNISTCritic
        h_pool = feat.sum(dim=[2, 3])           # [B, fd*8]
        emb = self.label_emb(y)                 # [B, fd*8]
        proj = (h_pool * emb).sum(dim=1).view(-1, 1, 1, 1)
        return base + proj


# ============================================================================
# CartoonSet experiment -- CLIP-conditioned, 64x64 RGB (Section 6.4-6.5 of
# the thesis). Three generator variants share `AdaGN`/`DecoderBlock`:
#   - ClipCondGeneratorAdaGN     (E16): CLIP injected at bottleneck + decoder only
#   - GeneratorAdaGNAttention    (E17): + self-attention at the 8x8 decoder stage
#   - ClipCondGeneratorFullyInjected (E19-E21, the architecture used for the
#     thesis's final reported CartoonSet results): CLIP also injected at every
#     *encoder* stage via `EncoderBlockAdaGN`, matching the AdaGN-in-encoder
#     description in the thesis's architecture table.
# ============================================================================

class AdaGN(nn.Module):
    """Adaptive Group Normalization: GroupNorm(x), then scale/shift predicted
    from a projection of the CLIP embedding -- so every feature map is
    conditioned on the text/image prompt.

    The projection is zero-initialized so AdaGN starts as plain GroupNorm
    (gamma=1, beta=0) and learns to deviate from there.
    """

    def __init__(self, num_channels, emb_dim=512, num_groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, num_channels * 2, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, emb):
        x = self.gn(x)
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        gamma = (gamma + 1).view(-1, x.size(1), 1, 1)  # +1 -> identity at init
        beta = beta.view(-1, x.size(1), 1, 1)
        return gamma * x + beta


class EncoderBlockAdaGN(nn.Module):
    """One downsampling encoder step: Conv -> [optional AdaGN] -> LeakyReLU.
    Used only by `ClipCondGeneratorFullyInjected`; the other two generator
    variants use plain (unconditioned) conv blocks in the encoder."""

    def __init__(self, in_channels, out_channels, clip_dim=512, inject_clip=True,
                 kernel=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.inject_clip = inject_clip
        if self.inject_clip:
            self.adagn = AdaGN(out_channels, emb_dim=clip_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, clip_emb):
        x = self.conv(x)
        if self.inject_clip:
            x = self.adagn(x, clip_emb)
        return self.act(x)


class DecoderBlock(nn.Module):
    """One upsampling decoder step:
      ConvTranspose -> [optional CLIP addition] -> AdaGN -> LeakyReLU -> [Dropout]

    kernel/stride/padding: use (4,2,1) for a x2 upsample, (2,1,0) for the
    special 1x1 -> 2x2 first decoder block.
    """

    def __init__(self, in_ch, out_ch, clip_dim=512,
                 inject_clip=False, dropout=0.0, num_groups=8,
                 kernel=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.inject_clip = inject_clip
        if inject_clip:
            self.clip_inj = nn.Linear(clip_dim, out_ch, bias=True)
        self.adagn = AdaGN(out_ch, clip_dim, num_groups)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, clip_emb):
        x = self.conv(x)
        if self.inject_clip:
            c = self.clip_inj(clip_emb).view(-1, x.size(1), 1, 1)
            x = x + c
        x = self.adagn(x, clip_emb)
        x = self.act(x)
        return self.drop(x)


class SpatialSelfAttention(nn.Module):
    """Standard self-attention over a 2D feature map. `gamma` starts at zero
    so the layer is an identity at initialization."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.q = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, ch, h, w = x.size()
        q = self.q(x).view(batch, -1, h * w).permute(0, 2, 1)  # [B, HW, C//8]
        k = self.k(x).view(batch, -1, h * w)                    # [B, C//8, HW]
        v = self.v(x).view(batch, -1, h * w)                    # [B, C, HW]

        attn_scores = torch.bmm(q, k) * ((ch // 8) ** -0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.bmm(v, attn_probs.permute(0, 2, 1))
        out = out.view(batch, ch, h, w)
        return x + self.gamma * self.out_conv(out)


class ClipCondGeneratorAdaGN(nn.Module):
    """64x64 U-Net generator (E16). CLIP is injected at the bottleneck (added
    directly to the encoder output) and in the decoder (via AdaGN in each
    `DecoderBlock`); the encoder itself has no conditioning."""

    def __init__(self, channels_img=3, features_g=128, clip_dim=512, ch_cap=512):
        super().__init__()
        f = features_g
        C = lambda n: min(n, ch_cap)

        self.null_token = nn.Parameter(torch.zeros(clip_dim))
        self.clip_proj = nn.Linear(clip_dim, C(f * 16), bias=True)

        self.enc1 = nn.Sequential(nn.Conv2d(channels_img, C(f), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(C(f), C(f * 2), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(C(f * 2), C(f * 4), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(C(f * 4), C(f * 8), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(nn.Conv2d(C(f * 8), C(f * 16), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc6 = nn.Sequential(
            nn.Conv2d(C(f * 16), C(f * 16), 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2), nn.Dropout2d(0.3))  # bottleneck

        self.dec1 = DecoderBlock(C(f * 16), C(f * 8), clip_dim, inject_clip=False,
                                  dropout=0.2, kernel=2, stride=1, padding=0)
        self.dec2 = DecoderBlock(C(f * 8) + C(f * 16), C(f * 8), clip_dim, inject_clip=True)
        self.dec3 = DecoderBlock(C(f * 8) + C(f * 8), C(f * 4), clip_dim, inject_clip=True)
        self.dec4 = DecoderBlock(C(f * 4) + C(f * 4), C(f * 2), clip_dim, inject_clip=True)
        self.dec5 = DecoderBlock(C(f * 2) + C(f * 2), C(f), clip_dim, inject_clip=False)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(C(f) + C(f), channels_img, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, x, clip_emb):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)  # bottleneck

        c = self.clip_proj(clip_emb).view(clip_emb.size(0), -1, 1, 1)
        e6 = e6 + c

        d1 = self.dec1(e6, clip_emb)
        d2 = self.dec2(torch.cat([d1, e5], dim=1), clip_emb)
        d3 = self.dec3(torch.cat([d2, e4], dim=1), clip_emb)
        d4 = self.dec4(torch.cat([d3, e3], dim=1), clip_emb)
        d5 = self.dec5(torch.cat([d4, e2], dim=1), clip_emb)
        return self.out(torch.cat([d5, e1], dim=1))


class GeneratorAdaGNAttention(nn.Module):
    """Same as `ClipCondGeneratorAdaGN` (E16), plus a single
    `SpatialSelfAttention` layer at the 8x8 decoder stage (E17)."""

    def __init__(self, channels_img=3, features_g=128, clip_dim=512, ch_cap=512):
        super().__init__()
        f = features_g
        C = lambda n: min(n, ch_cap)

        self.null_token = nn.Parameter(torch.zeros(clip_dim))
        self.clip_proj = nn.Linear(clip_dim, C(f * 16), bias=True)

        self.enc1 = nn.Sequential(nn.Conv2d(channels_img, C(f), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(C(f), C(f * 2), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(C(f * 2), C(f * 4), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(C(f * 4), C(f * 8), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc5 = nn.Sequential(nn.Conv2d(C(f * 8), C(f * 16), 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.enc6 = nn.Sequential(
            nn.Conv2d(C(f * 16), C(f * 16), 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2), nn.Dropout2d(0.3))

        self.dec1 = DecoderBlock(C(f * 16), C(f * 8), clip_dim, inject_clip=False,
                                  dropout=0.2, kernel=2, stride=1, padding=0)
        self.dec2 = DecoderBlock(C(f * 8) + C(f * 16), C(f * 8), clip_dim, inject_clip=True)
        self.dec3 = DecoderBlock(C(f * 8) + C(f * 8), C(f * 4), clip_dim, inject_clip=True)
        self.dec3_attn = SpatialSelfAttention(in_channels=C(f * 4))  # at 8x8
        self.dec4 = DecoderBlock(C(f * 4) + C(f * 4), C(f * 2), clip_dim, inject_clip=True)
        self.dec5 = DecoderBlock(C(f * 2) + C(f * 2), C(f), clip_dim, inject_clip=False)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(C(f) + C(f), channels_img, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, x, clip_emb):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        c = self.clip_proj(clip_emb).view(clip_emb.size(0), -1, 1, 1)
        e6 = e6 + c

        d1 = self.dec1(e6, clip_emb)
        d2 = self.dec2(torch.cat([d1, e5], dim=1), clip_emb)
        d3 = self.dec3(torch.cat([d2, e4], dim=1), clip_emb)
        d3 = self.dec3_attn(d3)
        d4 = self.dec4(torch.cat([d3, e3], dim=1), clip_emb)
        d5 = self.dec5(torch.cat([d4, e2], dim=1), clip_emb)
        return self.out(torch.cat([d5, e1], dim=1))


class ClipCondGeneratorFullyInjected(nn.Module):
    """64x64 U-Net generator (E19-E21) -- the architecture used for the
    thesis's final reported CartoonSet results. Unlike `ClipCondGeneratorAdaGN`,
    CLIP is injected via AdaGN in *every* encoder block (enc2-enc6, via
    `EncoderBlockAdaGN`) as well as the decoder. `enc1` still skips conditioning, to keep
    the raw noise/image distribution stable at the very first layer."""

    def __init__(self, channels_img=3, features_g=128, clip_dim=512, ch_cap=512):
        super().__init__()
        f = features_g
        C = lambda n: min(n, ch_cap)

        self.null_token = nn.Parameter(torch.zeros(clip_dim))
        self.clip_proj = nn.Linear(clip_dim, C(f * 16), bias=True)

        self.enc1 = EncoderBlockAdaGN(channels_img, C(f), clip_dim, inject_clip=False)
        self.enc2 = EncoderBlockAdaGN(C(f), C(f * 2), clip_dim, inject_clip=True)
        self.enc3 = EncoderBlockAdaGN(C(f * 2), C(f * 4), clip_dim, inject_clip=True)
        self.enc4 = EncoderBlockAdaGN(C(f * 4), C(f * 8), clip_dim, inject_clip=True)
        self.enc5 = EncoderBlockAdaGN(C(f * 8), C(f * 16), clip_dim, inject_clip=True)

        # enc6 (bottleneck): kernel=2, stride=1, padding=0
        self.enc6_conv = nn.Conv2d(C(f * 16), C(f * 16), 2, 1, 0, bias=False)
        self.enc6_adagn = AdaGN(C(f * 16), clip_dim)
        self.enc6_act = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout2d(0.3))

        self.dec1 = DecoderBlock(C(f * 16), C(f * 8), clip_dim, inject_clip=False,
                                  dropout=0.2, kernel=2, stride=1, padding=0)
        self.dec2 = DecoderBlock(C(f * 8) + C(f * 16), C(f * 8), clip_dim, inject_clip=True)
        self.dec3 = DecoderBlock(C(f * 8) + C(f * 8), C(f * 4), clip_dim, inject_clip=True)
        self.dec4 = DecoderBlock(C(f * 4) + C(f * 4), C(f * 2), clip_dim, inject_clip=True)
        self.dec5 = DecoderBlock(C(f * 2) + C(f * 2), C(f), clip_dim, inject_clip=False)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(C(f) + C(f), channels_img, 4, 2, 1, bias=False), nn.Sigmoid())

    def forward(self, x, clip_emb):
        e1 = self.enc1(x, clip_emb)
        e2 = self.enc2(e1, clip_emb)
        e3 = self.enc3(e2, clip_emb)
        e4 = self.enc4(e3, clip_emb)
        e5 = self.enc5(e4, clip_emb)

        e6 = self.enc6_conv(e5)
        e6 = self.enc6_adagn(e6, clip_emb)
        e6 = self.enc6_act(e6)

        c = self.clip_proj(clip_emb).view(clip_emb.size(0), -1, 1, 1)
        e6 = e6 + c

        d1 = self.dec1(e6, clip_emb)
        d2 = self.dec2(torch.cat([d1, e5], dim=1), clip_emb)
        d3 = self.dec3(torch.cat([d2, e4], dim=1), clip_emb)
        d4 = self.dec4(torch.cat([d3, e3], dim=1), clip_emb)
        d5 = self.dec5(torch.cat([d4, e2], dim=1), clip_emb)
        return self.out(torch.cat([d5, e1], dim=1))


class ClipProjCritic(nn.Module):
    """CartoonSet critic: projects the CLIP embedding to a spatial map and
    concatenates it as a 4th input channel (simpler than the generator's
    per-layer AdaGN conditioning -- the critic only needs to judge
    consistency with the prompt, not control fine-grained generation)."""

    def __init__(self, channels_img=3, features_d=128, clip_dim=512, ch_cap=512, img_size=64):
        super().__init__()
        f = features_d
        C = lambda n: min(n, ch_cap)
        self.img_size = img_size

        self.clip_proj = nn.Linear(clip_dim, img_size * img_size, bias=True)

        self.net = nn.Sequential(
            nn.Conv2d(channels_img + 1, C(f), 4, 2, 1),  # 64 -> 32; +1 = CLIP spatial map
            nn.LeakyReLU(0.2),
            nn.Conv2d(C(f), C(f * 2), 4, 2, 1, bias=False),  # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(C(f * 2), C(f * 4), 4, 2, 1, bias=False),  # 16 -> 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(C(f * 4), C(f * 8), 4, 2, 1, bias=False),  # 8 -> 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(C(f * 8), C(f * 16), 4, 2, 1, bias=False),  # 4 -> 2
            nn.LeakyReLU(0.2),
            nn.Conv2d(C(f * 16), 1, kernel_size=2, stride=1, padding=0),  # 2 -> 1
        )

    def forward(self, x, clip_emb):
        B = clip_emb.size(0)
        c_map = self.clip_proj(clip_emb).view(B, 1, self.img_size, self.img_size)
        return self.net(torch.cat([x, c_map], dim=1))
