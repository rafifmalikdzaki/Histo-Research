"""
GradCAM Implementation for DAE-KAN Model Interpretability

This module implements GradCAM specifically designed for autoencoder architectures
and provides tools to visualize and analyze attention patterns in histopathology images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

from models.model import DAE_KAN_Attention
from histodata import create_dataset, ImageDataset
from torch.utils.data import DataLoader


class AutoencoderGradCAM:
    """
    Custom GradCAM implementation for autoencoder architectures
    """

    def __init__(self, model: nn.Module, target_layers: List[nn.Module], use_cuda: bool = True):
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Hook to store activations and gradients
        self.activations = {}
        self.gradients = {}

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""

        def forward_hook(module, input, output):
            self.activations[module] = output.detach().float()

        def backward_hook(module, grad_input, grad_output):
            self.gradients[module] = grad_output[0].detach().float()

        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor,
                    target_layers: Optional[List[nn.Module]] = None) -> Dict[nn.Module, np.ndarray]:
        """
        Generate Class Activation Maps for target layers
        """
        if target_layers is None:
            target_layers = self.target_layers

        input_tensor = input_tensor.to(self.device)
        # Ensure input is float32 for GradCAM computation
        if input_tensor.dtype == torch.float16:
            input_tensor = input_tensor.float()
        input_tensor.requires_grad_(True)

        # Forward pass
        encoded, decoded, z = self.model(input_tensor)

        # Use reconstruction loss as the target
        loss = F.mse_loss(decoded, input_tensor)
        loss.backward()

        cams = {}

        for layer in target_layers:
            if layer in self.activations and layer in self.gradients:
                # Get activations and gradients
                activations = self.activations[layer]  # [B, C, H, W]
                gradients = self.gradients[layer]     # [B, C, H, W]

                # Ensure float32 for computations
                if activations.dtype == torch.float16:
                    activations = activations.float()
                if gradients.dtype == torch.float16:
                    gradients = gradients.float()

                # Global average pooling of gradients to get weights
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

                # Weighted combination of activation maps
                cam = torch.sum(weights * activations, dim=1)  # [B, H, W]

                # Apply ReLU to focus on positive contributions
                cam = F.relu(cam)

                # Normalize to [0, 1]
                cam_flat = cam.view(cam.size(0), -1)
                cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
                cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

                cams[layer] = cam.cpu().numpy()

        return cams

    def generate_multi_scale_cam(self, input_tensor: torch.Tensor,
                               scales: List[int] = [0.8, 1.0, 1.2]) -> Dict[str, np.ndarray]:
        """
        Generate multi-scale CAMs for better analysis
        """
        multi_scale_cams = {}

        for scale in scales:
            # Resize input
            orig_size = input_tensor.shape[-2:]
            new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
            resized_input = F.interpolate(input_tensor, size=new_size, mode='bilinear', align_corners=False)

            # Generate CAM
            cams = self.generate_cam(resized_input)

            # Resize CAMs back to original size
            for layer, cam in cams.items():
                resized_cam = F.interpolate(
                    torch.from_numpy(cam).unsqueeze(1),
                    size=orig_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                multi_scale_cams[f"{layer.__class__.__name__}_scale_{scale}"] = resized_cam

        return multi_scale_cams


class DAEKANAnalyzer:
    """
    Comprehensive analyzer for DAE-KAN model interpretability
    """

    def __init__(self, model: Optional[nn.Module] = None, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model
            # Ensure the model is on the correct device
            if not next(model.parameters()).is_cuda and "cuda" in device:
                self.model = model.to(self.device)
        else:
            self.model = DAE_KAN_Attention().to(self.device)

        if model_path:
            self.load_model(model_path)

        # Define target layers for GradCAM
        self.target_layers = [
            self.model.ae_encoder.encoder3,  # Bottleneck features
            self.model.bottleneck.attn1,      # First BAM attention
            self.model.bottleneck.attn2,      # Second BAM attention
        ]

        self.gradcam = AutoencoderGradCAM(self.model, self.target_layers, use_cuda=(device == "cuda"))

    def load_model(self, model_path: str):
        """Load pretrained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")

    def analyze_sample(self, input_image: torch.Tensor,
                      save_path: Optional[str] = None) -> Dict:
        """
        Comprehensive analysis of a single sample
        """
        self.model.eval()
        input_image = input_image.to(self.device)

        # Generate reconstructions
        with torch.no_grad():
            encoded, decoded, z = self.model(input_image)

        # Generate CAMs
        cams = self.gradcam.generate_cam(input_image)

        # Calculate reconstruction metrics
        mse_loss = F.mse_loss(decoded, input_image).item()
        ssim_score = self.calculate_ssim(input_image, decoded).item()

        # Extract attention weights from BAM layers
        attention_weights = self.extract_bam_attention(input_image)

        analysis_result = {
            'input_image': input_image.cpu().numpy(),
            'reconstructed_image': decoded.cpu().numpy(),
            'latent_representation': z.cpu().numpy(),
            'cams': cams,
            'attention_weights': attention_weights,
            'mse_loss': mse_loss,
            'ssim_score': ssim_score,
            'encoding_features': encoded.cpu().numpy()
        }

        if save_path:
            self.save_analysis_results(analysis_result, save_path)

        return analysis_result

    def extract_bam_attention(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract spatial and channel attention from BAM layers
        """
        attentions = {}

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                # BAM outputs attention-weighted features, convert to float32 for compatibility
                attentions[layer_name] = output.detach().float().cpu().numpy()
            return hook_fn

        # Register hooks for BAM layers
        hooks = []
        hook_layers = [
            ('attn1_pre', self.model.bottleneck.attn1),
            ('attn2_pre', self.model.bottleneck.attn2),
        ]

        for name, layer in hook_layers:
            hook = layer.register_forward_hook(create_hook_fn(name))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return attentions

    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Structural Similarity Index
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = torch.mean(img1, dim=[2, 3], keepdim=True)
        mu2 = torch.mean(img2, dim=[2, 3], keepdim=True)

        sigma1_sq = torch.var(img1, dim=[2, 3], keepdim=True, unbiased=False)
        sigma2_sq = torch.var(img2, dim=[2, 3], keepdim=True, unbiased=False)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[2, 3], keepdim=True)

        ssim_num = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        ssim_den = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim = ssim_num / ssim_den
        return torch.mean(ssim)

    def visualize_analysis(self, analysis_result: Dict,
                          save_path: Optional[str] = None):
        """
        Create comprehensive visualization of analysis results
        """
        fig = plt.figure(figsize=(20, 12))

        # Original image
        ax1 = plt.subplot(3, 4, 1)
        img = analysis_result['input_image'][0].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Reconstructed image
        ax2 = plt.subplot(3, 4, 2)
        recon = analysis_result['reconstructed_image'][0].transpose(1, 2, 0)
        recon = (recon - recon.min()) / (recon.max() - recon.min())
        ax2.imshow(recon)
        ax2.set_title(f'Reconstructed\nMSE: {analysis_result["mse_loss"]:.4f}\nSSIM: {analysis_result["ssim_score"]:.4f}')
        ax2.axis('off')

        # Residual (error) map
        ax3 = plt.subplot(3, 4, 3)
        residual = np.abs(img - recon)
        residual_mean = np.mean(residual, axis=2)
        im = ax3.imshow(residual_mean, cmap='hot')
        ax3.set_title('Reconstruction Error')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # Latent space visualization
        ax4 = plt.subplot(3, 4, 4)
        latent = analysis_result['latent_representation'][0]
        if latent.shape[0] > 1:
            # Visualize first few latent dimensions
            latent_viz = latent[:min(16, latent.shape[0])]
            latent_viz = latent_viz.reshape(-1, 8)
            im = ax4.imshow(latent_viz, cmap='viridis', aspect='auto')
            ax4.set_title('Latent Representation')
            ax4.set_xlabel('Latent Dimension')
            ax4.set_ylabel('Feature')
            plt.colorbar(im, ax=ax4, fraction=0.046)

        # CAMs for different layers
        cam_idx = 5
        for i, (layer, cam) in enumerate(analysis_result['cams'].items()):
            if cam_idx > 8:  # Limit to 4 CAM visualizations
                break

            ax = plt.subplot(3, 4, cam_idx)
            cam_img = cam[0]  # First image in batch
            im = ax.imshow(cam_img, cmap='jet')
            ax.set_title(f'CAM: {layer.__class__.__name__}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            cam_idx += 1

        # Attention visualizations
        for i, (name, attention) in enumerate(analysis_result['attention_weights'].items()):
            if cam_idx > 12:  # Limit visualizations
                break

            ax = plt.subplot(3, 4, cam_idx)
            attn_img = attention[0]
            if attn_img.shape[0] > 3:  # Multi-channel attention
                # Average across channels for visualization
                attn_viz = np.mean(attn_img, axis=0)
            else:
                attn_viz = attn_img[0]

            im = ax.imshow(attn_viz, cmap='viridis')
            ax.set_title(f'Attention: {name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            cam_idx += 1

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_analysis_results(self, analysis_result: Dict, save_path: str):
        """Save analysis results to disk"""
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(analysis_result, f)

    def batch_analyze(self, dataloader: DataLoader,
                     num_samples: int = 50,
                     save_dir: str = "../analysis/visualizations/"):
        """
        Analyze multiple samples from the dataset
        """
        self.model.eval()
        results = []

        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            x, y = batch
            if x.shape[0] > 1:  # Take single image from batch
                x = x[:1]
                y = y[:1]

            # Analyze sample
            result = self.analyze_sample(x)

            # Save visualization
            sample_save_path = f"{save_dir}/sample_{i:04d}_analysis.png"
            self.visualize_analysis(result, sample_save_path)

            # Save detailed results
            detail_save_path = f"{save_dir}/sample_{i:04d}_results.pkl"
            self.save_analysis_results(result, detail_save_path)

            results.append({
                'sample_id': i,
                'mse_loss': result['mse_loss'],
                'ssim_score': result['ssim_score'],
                'save_path': sample_save_path,
                'detail_path': detail_save_path
            })

            print(f"Analyzed sample {i+1}/{num_samples}: MSE={result['mse_loss']:.4f}, SSIM={result['ssim_score']:.4f}")

        # Save summary
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(f"{save_dir}/batch_analysis_summary.csv", index=False)

        return results


def main():
    """
    Main function to run GradCAM analysis
    """
    print("Initializing DAE-KAN GradCAM Analysis...")

    # Create analyzer
    analyzer = DAEKANAnalyzer(device="cuda")

    # Create dataset
    train_ds = ImageDataset(*create_dataset('train'))
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Analyze a few samples
    print("Analyzing samples...")
    results = analyzer.batch_analyze(dataloader, num_samples=10)

    print("\nGradCAM Analysis Complete!")
    print(f"Analyzed {len(results)} samples")
    print("Results saved to ../analysis/visualizations/")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()