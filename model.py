import torch
from torch import nn
from torch.nn import functional as F
# Load Pillow before torchvision when importing the model on its own.
from PIL import Image as _pillow_image
from torchvision.models import ResNet18_Weights, resnet18


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        mean = self.channel_mlp(x.mean(dim=(2, 3), keepdim=True))
        maximum = self.channel_mlp(x.amax(dim=(2, 3), keepdim=True))
        x = x * (mean + maximum).sigmoid()
        stats = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * self.spatial(stats).sigmoid()


class ResNetCBAM(nn.Module):
    def __init__(self, pretrained=True, cbam=True, dropout=0.0):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if cbam:
            backbone.layer4 = nn.Sequential(backbone.layer4[0], CBAM(512), backbone.layer4[1])
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(3 * 512, 3)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def encode_views(self, views):
        # Each input is [slices, 3, height, width] for the same patient.
        lengths = [view.shape[0] for view in views]
        slices = torch.cat(views, dim=0)
        features = self.backbone((slices - self.mean) / self.std)
        return torch.stack([part.max(dim=0).values for part in features.split(lengths)])

    def forward(self, views):
        pooled = self.encode_views(views).flatten()
        return self.classifier(self.dropout(pooled))[None]


class SIPResNetCBAM(ResNetCBAM):
    def __init__(self, pretrained=True, cbam=True, dropout=0.0, prototype_samples=10):
        super().__init__(pretrained=pretrained, cbam=cbam, dropout=dropout)
        if prototype_samples < 1:
            raise ValueError("prototype_samples must be positive")
        labels = self.classifier.out_features
        dim = self.backbone.layer4[-1].bn2.num_features
        self.prototype_samples = prototype_samples
        # SIP optimizes the label inputs, initialized to one-hot rows.
        self.label_inputs = nn.Parameter(torch.eye(labels))
        self.prototype_mean = nn.Linear(labels, dim)
        self.prototype_variance = nn.Linear(labels, dim)
        self.view_attention = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)
        self.classifier = nn.Linear(dim, labels)

    def disease_prototypes(self):
        mean = self.prototype_mean(self.label_inputs)
        if not self.training:
            return mean
        variance = F.softplus(self.prototype_variance(self.label_inputs)) + 1e-6
        noise = torch.randn((self.prototype_samples, *mean.shape), device=mean.device, dtype=mean.dtype)
        return (mean + variance.sqrt() * noise).mean(dim=0)

    def forward(self, views, *, return_aux=False):
        if len(views) != 3 or any(view.ndim != 4 or view.shape[0] == 0 or view.shape[1] != 3 for view in views):
            raise ValueError("Expected three nonempty views with shape [slices, 3, H, W]")
        tokens = self.encode_views(views)[None]  # [1, views, 512]
        prototypes = self.disease_prototypes()  # [labels, 512]
        fused, weights = self.view_attention(prototypes[None], tokens, tokens, need_weights=return_aux)
        # Each disease classifier uses only its corresponding fused feature.
        logits = (self.dropout(fused) * self.classifier.weight).sum(dim=-1) + self.classifier.bias
        if return_aux:
            return logits, prototypes, tokens.mean(dim=1), weights
        return logits


def prototype_aggregation_loss(targets, prototypes, global_features):
    positive = (targets == 1).to(prototypes.dtype)
    counts = positive.sum(dim=1, keepdim=True)
    centers = (positive @ prototypes) / counts.clamp_min(1)
    distances = (global_features - centers).square().sum(dim=1)
    valid = counts.squeeze(1) > 0
    return distances.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
