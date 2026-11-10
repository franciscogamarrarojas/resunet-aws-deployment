import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if in_channels != out_channels:

            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.residual_block = ResidualBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.residual_block(x)

        return x


class ResUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(
            1, 64
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(
            64, 128
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(
            128, 256
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualBlock(
            256, 512
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(
            512, 1024
        )

        # Decoder
        self.dec4 = DecoderBlock(
            1024, 512, 512
        )

        self.dec3 = DecoderBlock(
            512, 256, 256
        )

        self.dec2 = DecoderBlock(
            256, 128, 128
        )

        self.dec1 = DecoderBlock(
            128, 64, 64
        )

        # Output
        self.output_conv = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool4(e4)
        )

        # Decoder + skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Logits
        output = self.output_conv(d1)

        return output
