import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class FCN(nn.Module):
    def __init__(self, variant="fcn8s", num_classes=21):
        super().__init__()
        self.variant = variant.lower()

        # Load a VGG16 backbone with pretrained weights.
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())

        # Define feature extractors:
        # pool3: up to layer 16 (inclusive) produces features at 1/8 resolution.
        self.pool3 = nn.Sequential(*features[:17])
        # pool4: layers 17 to 23 produce features at 1/16 resolution.
        self.pool4 = nn.Sequential(*features[17:24])
        # pool5: layers 24 to 30 produce features at 1/32 resolution.
        self.pool5 = nn.Sequential(*features[24:31])

        # Score layers: 1x1 convolutions to map features to the desired number of classes.
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(512, num_classes, kernel_size=1)

        # Upsampling layer shared between branches.
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )

        # Final upsampling layer depends on the variant.
        if self.variant == "fcn32s":
            self.final_upsampler = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=64, stride=32, padding=16
            )
        elif self.variant == "fcn16s":
            self.final_upsampler = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=32, stride=16, padding=8
            )
        elif self.variant == "fcn8s":
            self.final_upsampler = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=16, stride=8, padding=4
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def freeze_backbone(self):
        for layer in [self.pool3, self.pool4, self.pool5]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Extract features from VGG16 backbone.
        pool3 = self.pool3(x)  # (Batch, 256, H/8, W/8)
        pool4 = self.pool4(pool3)  # (Batch, 512, H/16, W/16)
        pool5 = self.pool5(pool4)  # (Batch, 512, H/32, W/32)

        # Generate initial score map from the deepest layer.
        score5 = self.score_pool5(pool5)  # (Batch, num_classes, H/32, W/32)

        if self.variant == "fcn32s":
            # Upsample directly to input resolution.
            return self.final_upsampler(score5)

        # For FCN-16s: Fuse with pool4.
        score4 = self.score_pool4(pool4)  # (Batch, num_classes, H/16, W/16)
        upscore2 = self.upscore2(score5)  # Upsample score5 to H/16, W/16
        fused = score4 + upscore2

        if self.variant == "fcn16s":
            return self.final_upsampler(fused)

        # For FCN-8s: Also fuse with pool3.
        score3 = self.score_pool3(pool3)  # (Batch, num_classes, H/8, W/8)
        upscore2_2 = self.upscore2(fused)  # Upsample fused score to H/8, W/8
        fused_final = score3 + upscore2_2
        return self.final_upsampler(fused_final)
