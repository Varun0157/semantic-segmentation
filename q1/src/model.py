import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class FCN(nn.Module):
    def __init__(self, variant="fcn8s", num_classes=21):
        super().__init__()
        self.variant = variant.lower()

        # Feature backbone
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        # NOTE: understand what this is
        features = list(vgg.features.children())
        self.pool3 = nn.Sequential(*features[:17])  # Output: (256, 28, 28)
        self.pool4 = nn.Sequential(*features[17:24])  # Output: (512, 14, 14)
        self.pool5 = nn.Sequential(*features[24:31])  # Output: (512, 7, 7)

        # Classifiers
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool5 = nn.Conv2d(512, num_classes, 1)

        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, padding=1
        )

        match self.variant:
            case "fcn32s":
                self.final_upsampler = nn.ConvTranspose2d(
                    num_classes, num_classes, 64, stride=32, padding=16
                )
            case "fcn16s":
                self.final_upsampler = nn.ConvTranspose2d(
                    num_classes, num_classes, 32, stride=16, padding=8
                )
            case "fcn8s":
                self.final_upsampler = nn.ConvTranspose2d(
                    num_classes, num_classes, 16, stride=8, padding=4
                )
            case _:
                raise ValueError("Invalid variant")

    def freeze_backbone(self):
        for layer in [self.pool3, self.pool4, self.pool5]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Feature extraction
        pool3 = self.pool3(x)  # (B, 256, 28, 28)
        pool4 = self.pool4(pool3)  # (B, 512, 14, 14)
        pool5 = self.pool5(pool4)  # (B, 512, 7, 7)

        # Base scores
        score5 = self.score_pool5(pool5)  # (B, C, 7, 7)

        if self.variant == "fcn32s":
            return self.final_upsampler(score5)

        # FCN-16s/8s path
        score4 = self.score_pool4(pool4)  # (B, C, 14, 14)
        upscore2 = self.upscore2(score5)  # (B, C, 14, 14)
        combined = score4 + upscore2

        if self.variant == "fcn16s":
            return self.final_upsampler(combined)

        # FCN-8s path
        score3 = self.score_pool3(pool3)  # (B, C, 28, 28)
        upscore2_2 = self.upscore2(combined)  # (B, C, 28, 28)
        return self.final_upsampler(score3 + upscore2_2)
