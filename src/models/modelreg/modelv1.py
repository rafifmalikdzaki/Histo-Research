import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from attention_mechanisms.bam import BAM
from attention_mechanisms.eca import ECALayer
from kan_convolutional.KANLinear import KAN
from kan_convolutional.KANConv import KAN_Convolutional_Layer as KANCL


class Autoencoder_Encoder(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Autoencoder_Encoder, self).__init__()

        self.kan = KANCL(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
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

        self.ECA_Net = ECALayer(32)

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, 128, kernel_size=3, stride=2, padding=1, output_padding=1
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

        out = checkpoint(_kan_forward, out, use_reentrant=False)
        out = self.ECA_Net(out)
        out = self.decoder1(out) + residual2
        out = self.decoder2(out) + residual1

        return out, residual1, residual2


class Autoencoder_Decoder(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Autoencoder_Decoder, self).__init__()

        self.kan = KANCL(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
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
            nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(True),
        )

        self.ECA_Net = ECALayer(32)

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, 128, kernel_size=3, stride=2, padding=1, output_padding=1
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

        self.Output_Layer = nn.Sequential(
            nn.Conv2d(384, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(True),
        )

        self.reconstruction = KANCL(
            in_channels=3,
            out_channels=3,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=3,
            spline_order=2,
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

        out = checkpoint(_kan_forward, out, use_reentrant=False)
        out = self.ECA_Net(out)

        # Simplified decoder without complex residual connections
        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.Output_Layer(out)
        out = checkpoint(_reconstruction_forward, out, use_reentrant=False)

        return out


class Autoencoder_Bottleneck(nn.Module):
    def __init__(self):
        super(Autoencoder_Bottleneck, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ELU(True),
        )

        self.attn1 = BAM(192)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(True),
        )

        self.attn2 = BAM(64)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ELU(True),
        )

    def forward(self, x: torch.Tensor):
        def _attn1_forward(x):
            return self.attn1(x)

        def _attn2_forward(x):
            return self.attn2(x)

        x = self.encoder1(x)
        x = checkpoint(_attn1_forward, x, use_reentrant=False)
        x = self.encoder2(x)
        z = checkpoint(_attn2_forward, x, use_reentrant=False)
        x = self.decoder(z)

        return x, z


class DAE_KAN_Attention(nn.Module):
    def __init__(self):
        super(DAE_KAN_Attention, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ae_encoder = Autoencoder_Encoder(device=self.device)
        self.bottleneck = Autoencoder_Bottleneck()
        self.ae_decoder = Autoencoder_Decoder(device=self.device)

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
    x_b = torch.randn(1, 384, 16, 16)
    model_en = Autoencoder_Encoder(device="cuda").to("cuda")
    model_de = Autoencoder_Decoder(device="cuda").to("cuda")
    model_bn = Autoencoder_Bottleneck().to("cuda")
    complete = DAE_KAN_Attention().to("cuda")
    complete.eval()
    with torch.no_grad():
        x, y, z = complete(x_e)

    # summary(complete, x_e)
    print("Success! Model ran without errors.")
    print(f"Input shape: {x_e.shape}")
    print(f"Encoded shape: {x.shape}")
    print(f"Decoded shape: {y.shape}")
    print(f"Latent shape: {z.shape}")
