import torch
from torch import nn
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

    def forward(self, views):
        # Each input is [slices, 3, height, width] for the same patient.
        lengths = [view.shape[0] for view in views]
        slices = torch.cat(views, dim=0)
        features = self.backbone((slices - self.mean) / self.std)
        pooled = torch.cat([part.max(dim=0).values for part in features.split(lengths)])
        return self.classifier(self.dropout(pooled))[None]
