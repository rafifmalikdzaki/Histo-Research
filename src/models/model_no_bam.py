import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .kan_convolutional.efficient_kan import KAN
from .kan_convolutional.efficient_kan_conv_fixed import KAN_Convolutional_Layer as KANCL


class NoBAMAutoencoder_Encoder(nn.Module):
    """Encoder with KAN layers but no BAM attention"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(NoBAMAutoencoder_Encoder, self).__init__()

        self.kan = KANCL(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            grid_size=3,
            spline_order=2,
            device=device,
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
        )

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 384, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ELU(True),
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


class NoBAMAutoencoder_Decoder(nn.Module):
    """Decoder with KAN layers but no BAM attention"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(NoBAMAutoencoder_Decoder, self).__init__()

        self.kan = KANCL(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            grid_size=3,
            spline_order=2,
            device=device,
        )

        self.encoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
        )

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 384, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ELU(True),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(
                384, 384, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ELU(True),
        )

        self.Output_Layer = nn.Sequential(
            nn.Conv2d(384, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(True),
        )

        self.reconstruction = KANCL(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            padding=0,
            grid_size=2,
            spline_order=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
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


class NoBAMAutoencoder_Bottleneck(nn.Module):
    """Bottleneck with NO BAM attention - replaced with standard layers"""

    def __init__(self):
        super(NoBAMAutoencoder_Bottleneck, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(True),
        )

        # Replace BAM with standard attention-like layer
        self.attn1 = nn.Sequential(
            nn.Conv2d(384, 384 // 8, kernel_size=1),
            nn.BatchNorm2d(384 // 8),
            nn.ELU(True),
            nn.Conv2d(384 // 8, 384, kernel_size=1),
            nn.Sigmoid()
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(True),
        )

        # Replace BAM with standard attention-like layer
        self.attn2 = nn.Sequential(
            nn.Conv2d(16, 16 // 8, kernel_size=1),
            nn.BatchNorm2d(16 // 8),
            nn.ELU(True),
            nn.Conv2d(16 // 8, 16, kernel_size=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

    def forward(self, x: torch.Tensor):
        def _attn1_forward(x):
            # Simple attention mechanism
            attn_weights = self.attn1(x)
            return x * attn_weights

        def _attn2_forward(x):
            # Simple attention mechanism
            attn_weights = self.attn2(x)
            return x * attn_weights

        x = self.encoder1(x)
        x = _attn1_forward(x)
        x = self.encoder2(x)
        z = _attn2_forward(x)
        x = self.decoder(z)

        return x, z


class NoBAM_DAE_KAN_Attention(nn.Module):
    """
    DAE model WITH KAN layers but NO BAM attention:
    - KAN layers preserved in encoder/decoder
    - BAM attention replaced with standard attention mechanisms
    """

    def __init__(self):
        super(NoBAM_DAE_KAN_Attention, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ae_encoder = NoBAMAutoencoder_Encoder(device=self.device)
        self.bottleneck = NoBAMAutoencoder_Bottleneck()
        self.ae_decoder = NoBAMAutoencoder_Decoder(device=self.device)

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
    model_en = NoBAMAutoencoder_Encoder(device="cuda").to("cuda")
    model_de = NoBAMAutoencoder_Decoder(device="cuda").to("cuda")
    model_bn = NoBAMAutoencoder_Bottleneck().to("cuda")
    complete = NoBAM_DAE_KAN_Attention().to("cuda")
    complete.eval()
    with torch.no_grad():
        x, y, z = complete(x_e)

    print("No-BAM model test successful!")
    print(f"Input shape: {x_e.shape}")
    print(f"Encoded shape: {x.shape}")
    print(f"Decoded shape: {y.shape}")
    print(f"Latent shape: {z.shape}")