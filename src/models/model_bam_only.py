import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .attention_mechanisms.bam import BAM


class BAMOnlyAutoencoder_Encoder(nn.Module):
    """Encoder with BAM attention only - no KAN layers"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(BAMOnlyAutoencoder_Encoder, self).__init__()

        # Replace KAN with BAM-enhanced standard convolution
        self.kan = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            BAM(64),  # Add BAM attention here
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(True),
            BAM(384),  # Add BAM attention
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            BAM(128),  # Add BAM attention
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            BAM(64),  # Add BAM attention
        )

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            BAM(128),  # Add BAM attention
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 384, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ELU(True),
            BAM(384),  # Add BAM attention
        )

    def forward(self, x: torch.Tensor):
        def _kan_forward(x):
            return self.kan(x)

        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)

        residual1 = self.encoder1(x)
        residual2 = self.encoder2(residual1)

        out = _kan_forward(out)
        out = self.decoder1(out) + residual2
        out = self.decoder2(out) + residual1

        return out, residual1, residual2


class BAMOnlyAutoencoder_Decoder(nn.Module):
    """Decoder with BAM attention only - no KAN layers"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(BAMOnlyAutoencoder_Decoder, self).__init__()

        # Replace KAN with BAM-enhanced standard convolution
        self.kan = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            BAM(64),  # Add BAM attention
        )

        self.encoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            BAM(128),  # Add BAM attention
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            BAM(128),  # Add BAM attention
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            BAM(64),  # Add BAM attention
        )

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            BAM(128),  # Add BAM attention
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 384, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ELU(True),
            BAM(384),  # Add BAM attention
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(
                384, 384, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ELU(True),
            BAM(384),  # Add BAM attention
        )

        self.Output_Layer = nn.Sequential(
            nn.Conv2d(384, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(True),
        )

        # Replace KAN reconstruction with BAM-enhanced standard convolution
        self.reconstruction = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ELU(True),
            BAM(3),  # Add BAM attention
        )

    def forward(
        self, x: torch.Tensor, residualEnc1: torch.Tensor, residualEnc2: torch.Tensor
    ):
        def _kan_forward(x):
            return self.kan(x)

        def _reconstruction_forward(x):
            return self.reconstruction(x)

        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)

        out = _kan_forward(out)

        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)
        out = self.Output_Layer(out)
        out = _reconstruction_forward(out)

        return out


class BAMOnlyAutoencoder_Bottleneck(nn.Module):
    """Bottleneck with enhanced BAM attention"""

    def __init__(self):
        super(BAMOnlyAutoencoder_Bottleneck, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(True),
        )

        # Enhanced BAM attention
        self.attn1 = BAM(384)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(True),
        )

        # Enhanced BAM attention
        self.attn2 = BAM(16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            BAM(128),  # Add BAM attention in decoder too
        )

    def forward(self, x: torch.Tensor):
        def _attn1_forward(x):
            return self.attn1(x)

        def _attn2_forward(x):
            return self.attn2(x)

        x = self.encoder1(x)
        x = _attn1_forward(x)
        x = self.encoder2(x)
        z = _attn2_forward(x)
        x = self.decoder(z)

        return x, z


class BAMOnly_DAE_KAN_Attention(nn.Module):
    """
    DAE model with BAM attention ONLY - no KAN layers:
    - All KAN layers replaced with standard convolutions
    - BAM attention added throughout the network
    - Pure attention-based autoencoder
    """

    def __init__(self):
        super(BAMOnly_DAE_KAN_Attention, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ae_encoder = BAMOnlyAutoencoder_Encoder(device=self.device)
        self.bottleneck = BAMOnlyAutoencoder_Bottleneck()
        self.ae_decoder = BAMOnlyAutoencoder_Decoder(device=self.device)

    def forward(self, x: torch.Tensor):
        encoded, residual1, residual2 = self.ae_encoder(x)
        decoded, z = self.bottleneck(encoded)
        decoded = self.ae_decoder(decoded, residual1, residual2)

        return encoded, decoded, z


if __name__ == "__main__":
    torch.cuda.empty_cache()
    from torchsummary import summary

    x_d = torch.randn(1, 128, 128, 128)
    x_e = torch.randn(1, 3, 128, 128).to("cuda")
    x_b = torch.randn(1, 384, 32, 32)
    model_en = BAMOnlyAutoencoder_Encoder(device="cuda").to("cuda")
    model_de = BAMOnlyAutoencoder_Decoder(device="cuda").to("cuda")
    model_bn = BAMOnlyAutoencoder_Bottleneck().to("cuda")
    complete = BAMOnly_DAE_KAN_Attention().to("cuda")
    complete.eval()
    with torch.no_grad():
        x, y, z = complete(x_e)

    print("BAM-Only model test successful!")
    print(f"Input shape: {x_e.shape}")
    print(f"Encoded shape: {x.shape}")
    print(f"Decoded shape: {y.shape}")
    print(f"Latent shape: {z.shape}")